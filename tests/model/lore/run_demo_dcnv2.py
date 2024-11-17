#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：run_demo_dcnv2
# @Author  ：cycloneboy
# @Date    ：20xx/10/26 18:27
import torch
from torchvision.ops import deform_conv2d


class RunDemoDcnv2(object):

    def __init__(self):
        pass

    def run(self):
        pass

    def dcn_demo(self):
        input = torch.rand(1, 512, 32, 32)
        kh, kw = 3, 3
        weight = torch.rand(256, 512, kh, kw)
        bias = torch.rand(256)
         # offset and mask should have the same spatial size as the output
         # of the convolution. In this case, for an input of 10, stride of 1
         # and kernel size of 3, without padding, the output size is 8
        offset = torch.rand(1, 2 * kh * kw, 32, 32)
        mask = torch.rand(1, kh * kw, 32, 32)
        out = deform_conv2d(input, offset=offset, weight=weight,
                            bias=bias,
                            stride=(1,1),
                            padding=(1,1),
                            dilation=(1,1),
                            mask=mask)
        print(out.shape)


def main():
    runner = RunDemoDcnv2()
    # runner.run()
    runner.dcn_demo()


if __name__ == '__main__':
    main()