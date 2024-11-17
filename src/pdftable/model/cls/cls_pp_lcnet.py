# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Dropout

MODEL_URLS = {
    "PPLCNet_x0.25":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_25_pretrained.pdparams",
    "PPLCNet_x0.35":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_35_pretrained.pdparams",
    "PPLCNet_x0.5":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_5_pretrained.pdparams",
    "PPLCNet_x0.75":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x0_75_pretrained.pdparams",
    "PPLCNet_x1.0":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_0_pretrained.pdparams",
    "PPLCNet_x1.5":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x1_5_pretrained.pdparams",
    "PPLCNet_x2.0":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_0_pretrained.pdparams",
    "PPLCNet_x2.5":
        "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/PPLCNet_x2_5_pretrained.pdparams"
}

MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

__all__ = list(MODEL_URLS.keys())

# Each element(list) represents a depthwise block, which is composed of k, in_c, out_c, s, use_se.
# k: kernel_size
# in_c: input channel number in depthwise block
# out_c: output channel number in depthwise block
# s: stride in depthwise block
# use_se: whether to use SE block

NET_CONFIG = {
    "blocks2":
    # k, in_c, out_c, s, use_se
        [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5":
        [[3, 128, 256, 2, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False],
         [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 num_groups=1):
        super().__init__()

        self.conv = Conv2d(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            bias=False)

        self.bn = BatchNorm2d(num_filters, track_running_stats=True)
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hardswish(x)
        return x


class DepthwiseSeparable(nn.Module):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 dw_size=3,
                 use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels)
        if use_se:
            self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.conv1 = Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.relu = nn.ReLU()
        self.conv2 = Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = torch.multiply(identity, x)
        return x


class PPLCNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 scale=1.0,
                 pretrained=False,
                 use_ssld=False,
                 class_num=1000,
                 dropout_prob=0.2,
                 class_expand=1280,
                 use_last_conv=True,
                 stride_list=[2, 2, 2, 2, 2],
                 **kwargs):
        super().__init__()
        self.out_channels = [
            int(NET_CONFIG["blocks3"][-1][2] * scale),
            int(NET_CONFIG["blocks4"][-1][2] * scale),
            int(NET_CONFIG["blocks5"][-1][2] * scale),
            int(NET_CONFIG["blocks6"][-1][2] * scale)
        ]
        self.scale = scale
        self.class_expand = class_expand
        self.use_last_conv = use_last_conv
        self.stride_list = stride_list
        self.net_config = NET_CONFIG

        for i, stride in enumerate(stride_list[1:]):
            self.net_config["blocks{}".format(i + 3)][0][3] = stride

        self.conv1 = ConvBNLayer(
            num_channels=in_channels,
            filter_size=3,
            num_filters=make_divisible(16 * scale),
            stride=2)

        self.blocks2 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks2"])
        ])

        self.blocks3 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks3"])
        ])

        self.blocks4 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks4"])
        ])

        self.blocks5 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks5"])
        ])

        self.blocks6 = nn.Sequential(*[
            DepthwiseSeparable(
                num_channels=make_divisible(in_c * scale),
                num_filters=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se)
            for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks6"])
        ])

        self.avg_pool = AdaptiveAvgPool2d(1)
        if self.use_last_conv:
            self.last_conv = Conv2d(
                in_channels=make_divisible(NET_CONFIG["blocks6"][-1][2] *
                                           scale),
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False)
            self.hardswish = nn.Hardswish()
            self.dropout = Dropout(p=dropout_prob)
        else:
            self.last_conv = None
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(
            self.class_expand if self.use_last_conv else
            make_divisible(NET_CONFIG["blocks6"][-1][2] * scale),
            class_num)

        if pretrained:
            self._load_pretrained(
                MODEL_URLS['PPLCNet_x{}'.format(scale)], use_ssld=use_ssld)

    def forward(self, image):
        outs = []
        x = self.conv1(image)
        x = self.blocks2(x)
        x = self.blocks3(x)
        outs.append(x)
        x = self.blocks4(x)
        outs.append(x)
        x = self.blocks5(x)
        outs.append(x)
        x = self.blocks6(x)
        outs.append(x)

        x = self.avg_pool(x)
        if self.last_conv is not None:
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        outs.append(x)
        return x

    def _load_pretrained(self, pretrained_url, use_ssld=False):
        if use_ssld:
            pretrained_url = pretrained_url.replace("_pretrained",
                                                    "_ssld_pretrained")
        print(pretrained_url)

        # TODO: 下载并转换模型
        # local_weight_path = get_path_from_url(
        #     pretrained_url, os.path.expanduser("~/.paddleclas/weights"))
        # param_state_dict = torch.load(local_weight_path)
        # self.load_state_dict(param_state_dict)
        return


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )



def PPLCNet_x0_25(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_25
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_25` model depends on args.
    """
    model = PPLCNet(
        scale=0.25, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_25"], use_ssld)
    return model


def PPLCNet_x0_35(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_35
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_35` model depends on args.
    """
    model = PPLCNet(
        scale=0.35, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_35"], use_ssld)
    return model


def PPLCNet_x0_5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_5` model depends on args.
    """
    model = PPLCNet(
        scale=0.5, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_5"], use_ssld)
    return model


def PPLCNet_x0_75(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x0_75
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x0_75` model depends on args.
    """
    model = PPLCNet(
        scale=0.75, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x0_75"], use_ssld)
    return model


def PPLCNet_x1_0(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x1_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x1_0` model depends on args.
    """
    model = PPLCNet(
        scale=1.0, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x1_0"], use_ssld)
    return model


def PPLCNet_x1_5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x1_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x1_5` model depends on args.
    """
    model = PPLCNet(
        scale=1.5, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x1_5"], use_ssld)
    return model


def PPLCNet_x2_0(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x2_0
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x2_0` model depends on args.
    """
    model = PPLCNet(
        scale=2.0, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x2_0"], use_ssld)
    return model


def PPLCNet_x2_5(pretrained=False, use_ssld=False, **kwargs):
    """
    PPLCNet_x2_5
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `PPLCNet_x2_5` model depends on args.
    """
    model = PPLCNet(
        scale=2.5, stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNet_x2_5"], use_ssld)
    return model
