#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable
# @File    ：run_ocr_convertvit
# @Author  ：cycloneboy
# @Date    ：20xx/4/4 15:10
import traceback

import torch
import torch.nn.functional as F
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from pdftable.model import OCRConvNextViTPreprocessor, ConvNextViT
from pdftable.utils import FileUtils, logger, Constants, CommonUtils


class RunTestConvertVit(object):

    def __init__(self):
        self.scope_model_base_dir = Constants.SCOPE_MODEL_BASE_DIR
        self.model_name_or_path = f"{self.scope_model_base_dir}/damo/cv_convnextTiny_ocr-recognition-general_damo"
        # self.fp16_full_eval = True
        self.fp16_full_eval = False
        self.do_transform = True
        self._init_class = "ConvNextViT"
        self.device = "cuda"
        self.img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'

        self.processor = OCRConvNextViTPreprocessor()

        self.labels = FileUtils.read_to_text_list(f"{self.model_name_or_path}/vocab.txt")
        self.labelMapping = {index + 2: item for index, item in enumerate(self.labels)}

        self.model_dict = {
            "general_damo": {
                "model": 'damo/cv_convnextTiny_ocr-recognition-general_damo',
                "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
            },
            "handwritten_damo": {
                "model": 'damo/cv_convnextTiny_ocr-recognition-handwritten_damo',
                "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_handwritten.jpg'
            },
            "document_damo": {
                "model": 'damo/cv_convnextTiny_ocr-recognition-document_damo',
                "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_document.png'
            },
            "licenseplate_damo": {
                "model": 'damo/cv_convnextTiny_ocr-recognition-licenseplate_damo',
                "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_licenseplate//ocr_recognition_licenseplate.jpg'
            },
            "scene_damo": {
                "model": 'damo/cv_convnextTiny_ocr-recognition-scene_damo',
                "image_url": 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
            },
        }

    def run(self):
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
        result = ocr_recognition(self.img_url)
        print(result)

    def ocr_scope_run(self):
        """
        scope run

        :return:
        """
        for key, val in self.model_dict.items():
            model = val["model"]
            img_url = val["image_url"]
            result = self.scope_model_predict(model, img_url)
            print(f"{key} - {model} - {result}")

    def ocr_run_compare(self):
        """
        scope run

        :return:
        """
        for key, val in self.model_dict.items():
            model = val["model"]
            img_url = val["image_url"]
            scope_result = self.scope_model_predict(model=model, img_url=img_url)
            logger.info(f"scope_result: {key} - {model} - {scope_result}")

            model_name_or_path = f"{self.scope_model_base_dir}/{model}"
            our_result = self.our_run_model(model_name_or_path=model_name_or_path, img_url=img_url)
            logger.info(f"our_result: {key} - {model} - {our_result}")

            if scope_result != our_result:
                logger.info(f"{model} predict diff: {scope_result} - {our_result}")
            else:
                logger.info(f"{model} predict same: {scope_result} - {our_result}")

    def scope_model_predict(self, model, img_url):
        """
        scope model predict

        :param model:
        :param img_url:
        :return:
        """
        ocr_recognition = pipeline(Tasks.ocr_recognition, model=model)
        result = ocr_recognition(img_url)

        predict = result["text"]
        logger.info(f"scope_model_predict: {predict}")

        ### 使用图像文件
        ### 请准备好名为'ocr_recognition.jpg'的图像文件
        # img_path = 'ocr_recognition.jpg'
        # img = cv2.imread(img_path)
        # result = ocr_recognition(img)
        # print(result)

        return predict

    def print_model(self):
        ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
        model = ocr_recognition.model

        save_dir = f"{Constants.DATA_DIR}/txt/model_scope/convnext_tiny_ocr/convnext_tiny_ocr"
        model_net, model_params = CommonUtils.print_model_param(model, save_dir=save_dir, use_numpy=True)

        model_our = ConvNextViT()
        save_dir2 = f"{Constants.DATA_DIR}/txt/model_scope/convnext_tiny_ocr/convnext_tiny_ocr_v2"
        model_net2, model_params2 = CommonUtils.print_model_param(model_our, save_dir=save_dir2, use_numpy=True)

        print(f"model")

    def convert_params(self, model_name_or_path=None):
        """
        转换模型参数

        :return:
        """
        if model_name_or_path is None:
            model_name_or_path = self.model_name_or_path

        pretrain_weight = f"{model_name_or_path}/pytorch_model.pt"
        output_dir = f"{Constants.DATA_DIR}/txt/model_scope/ocr/{self._init_class}/{FileUtils.get_parent_dir_name(pretrain_weight)}"

        model = ConvNextViT()
        pretrain_weight_new = load_pretrain_weight(model, pretrain_weight,
                                                   show_info=True,
                                                   do_transform=self.do_transform,
                                                   output_dir=output_dir,
                                                   model_name="ConvNextViT")

    def batch_convert_params(self):
        """
        batch convert params

        :return:
        """
        for key, val in self.model_dict.items():
            model = val["model"]
            model_name_or_path = f"{self.scope_model_base_dir}/{model}"
            try:
                self.convert_params(model_name_or_path=model_name_or_path)
            except Exception as e:
                traceback.print_exc()
            print(f"{key} - {model} - {model_name_or_path}")

    def our_run_model(self, model_name_or_path=None, img_url=None):
        if model_name_or_path is None:
            model_name_or_path = self.model_name_or_path
        model = ConvNextViT()
        model_path = f"{model_name_or_path}/pytorch_model.bin"
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        model.to(self.device)
        if self.fp16_full_eval:
            model.half()
        model.eval()

        if img_url is None:
            img_url = self.img_url

        batch = self.processor(img_url)
        batch = batch.to(self.device).contiguous()
        if self.fp16_full_eval:
            batch.half()

        print(f"{batch.shape}")

        with torch.no_grad():
            outputs = model(batch)

        print(f"outputs:{outputs}")

        logits = outputs.logits
        print(f"logits:{logits.shape}")
        out_preds = self.postprocess(logits)

        preds = out_preds['preds']
        print(f"preds:{preds}")

        return preds

    def postprocess(self, inputs):
        outprobs = inputs
        outprobs = F.softmax(outprobs, dim=-1)
        preds = torch.argmax(outprobs, -1)
        batchSize, length = preds.shape
        final_str_list = []
        for i in range(batchSize):
            pred_idx = preds[i].cpu().data.tolist()
            last_p = 0
            str_pred = []
            for p in pred_idx:
                if p != last_p and p != 0:
                    str_pred.append(self.labelMapping[p])
                last_p = p
            final_str = ''.join(str_pred)
            final_str_list.append(final_str)
        return {'preds': final_str_list, 'probs': inputs}


def main():
    runner = RunTestConvertVit()
    # runner.run()
    # runner.print_model()
    runner.convert_params()
    # runner.our_run_model()
    # runner.ocr_scope_run()
    # runner.batch_convert_params()
    # runner.ocr_run_compare()


if __name__ == '__main__':
    main()
