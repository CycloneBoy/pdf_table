#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project ：PdfTable 
# @File    ：wtw_dataset
# @Author  ：cycloneboy
# @Date    ：20xx/11/6 14:15
import os
from dataclasses import dataclass
from typing import Tuple, List, Any, Dict, Union

import numpy as np
import torch
from pycocotools import coco
from torch.utils.data import Dataset
from tqdm import tqdm

from pdftable.dataset.table.lore_coco_utils import LoreCocoUtils
from pdftable.entity import TableUnit, TableEval
from pdftable.model import TableLorePreProcessor, TableProcessUtils
from pdftable.model.lore.configuration_lore import LoreConfig
from pdftable.utils import logger, FileUtils


class TableWtw(object):

    def __init__(self, table_unit: List[TableUnit]):
        self.table_unit = table_unit

    def get_table_html(self):
        table_cells = TableProcessUtils.build_table_cell_from_table_unit(self.table_unit)
        table_html, db_table_html = TableProcessUtils.cell_to_html(table_cells=table_cells, first_header=False)
        return table_html

    def to_json(self):
        target = {}
        bboxes = []
        axis = []
        for row in self.table_unit:
            item = row.to_json()
            bboxes.append(item["bbox"])
            axis.append(item["axis"])

        target["bboxes"] = torch.from_numpy(np.array(bboxes))
        target["axis"] = torch.from_numpy(np.array(axis))
        # raw_html = self.get_table_html()
        # html = "\n".join(raw_html)
        # target["html"] = html

        return target


class WtwDataset(Dataset):

    def __init__(self, image_path, label_path=None, task_type="wtw", load_all=False,
                 split='train', txt_file=None
                 ):
        super().__init__()
        self.image_path = image_path
        self.label_path = label_path
        self.task_type = task_type
        self.load_all = load_all
        self.split = split
        self.txt_file = txt_file

        self.config = LoreConfig(task_type=self.task_type)
        self.pre_processor = TableLorePreProcessor(self.config)
        self.raw_images = list(sorted(os.listdir(image_path)))

        self.coco_data = None
        self.coco_image_id = None
        self.img_id_mapping = {}

        self.images, self.labels, self.coco_image_id = self.load_label(label_path)
        self.label_utils = LoreCocoUtils(config=self.config, split=split)

    def load_label(self, label_path) -> Tuple[List, Any, List]:
        if label_path is None:
            return self.raw_images, None, []

        logger.info(f"load label from {label_path}")
        coco_image_ids = []
        if label_path.endswith(".json"):
            exist_images, table_labels, coco_image_ids = self.load_coco_label(label_path)
        else:
            exist_images, table_labels = self.load_txt_label(label_path)

        if self.txt_file is not None:
            return self.load_sub_dataset(exist_images=exist_images,
                                         table_labels=table_labels,
                                         coco_image_ids=coco_image_ids)

        return exist_images, table_labels, coco_image_ids

    def load_sub_dataset(self, exist_images: List, table_labels: List, coco_image_ids: List):
        sub_exist_images = []
        sub_table_labels = []
        sub_coco_image_ids = []

        if isinstance(self.txt_file, int):
            skip = self.txt_file
            sub_exist_images = exist_images[skip:]
            # sub_table_labels = table_labels[skip:]
            sub_coco_image_ids = coco_image_ids[skip:]
            logger.info(f"skip image: {skip} - remain: {len(sub_exist_images)}")
        elif isinstance(self.txt_file, str) or (isinstance(self.txt_file, list) and isinstance(self.txt_file[0], str)):
            if isinstance(self.txt_file, str):
                image_file_names = FileUtils.read_to_text_list(self.txt_file)
            else:
                image_file_names = self.txt_file
            for index, name in enumerate(exist_images):
                if name in image_file_names:
                    sub_exist_images.append(name)
                    # sub_table_labels.append(table_labels[index])
                    sub_coco_image_ids.append(coco_image_ids[index])
            logger.info(f"skip image: {len(exist_images) - len(image_file_names)} - remain: {len(sub_exist_images)}")
        elif isinstance(self.txt_file, list) and isinstance(self.txt_file[0], int):
            image_index = self.txt_file
            for index in image_index:
                sub_exist_images.append(exist_images[index])
                # sub_table_labels.append(table_labels[index])
                sub_coco_image_ids.append(coco_image_ids[index])
            logger.info(f"skip image: {len(exist_images) - len(image_index)} - remain: {len(sub_exist_images)}")
        return sub_exist_images, sub_table_labels, sub_coco_image_ids

    def load_coco_label(self, label_path):
        self.coco_data = coco.COCO(label_path)
        images = self.coco_data.getImgIds()
        dir_image_set = set(self.raw_images)

        not_exist = 0
        exist_images = []
        exist_image_id = []
        table_labels = []
        for i in tqdm(range(len(images)), desc="load_table"):
            img_id = images[i]
            file_name = self.coco_data.loadImgs(ids=[img_id])[0]['file_name']
            if file_name not in dir_image_set:
                not_exist += 1
                continue

            if self.load_all:
                table = self.load_one_label_from_coco(img_id)
                table_labels.append(table)

            exist_image_id.append(img_id)
            self.img_id_mapping[img_id] = file_name

            # target = table.to_json()

            exist_images.append(file_name)

        logger.info(f"load label from coco finish, filter not label: {not_exist} , total: {len(exist_images)}")
        FileUtils.dump_json(label_path.replace(".json", "_image_id_mapping.json"), self.img_id_mapping)

        return exist_images, table_labels, exist_image_id

    def load_txt_label(self, label_path):
        dir_image_set = set(self.raw_images)
        not_exist = 0
        exist_images = []
        table_labels = []

        gt_bbox_path = os.path.join(label_path, 'gt_center')
        gt_logi_path = os.path.join(label_path, 'gt_logi')

        for file_name in tqdm(os.listdir(gt_bbox_path), desc="load_table"):
            raw_file_name = file_name[:-4]
            logi_path = os.path.join(gt_logi_path, file_name)
            if (raw_file_name not in dir_image_set or not file_name.endswith(".txt")
                    or not os.path.exists(logi_path)):
                not_exist += 1
                continue

            if self.load_all:
                table = self.load_one_label_from_txt(raw_file_name)
                table_labels.append(table)

            exist_images.append(raw_file_name)
        logger.info(f"load label from txt finish, filter not label: {not_exist} , total: {len(exist_images)}")
        return exist_images, table_labels

    def load_one_label_from_txt(self, file_name) -> TableWtw:
        bbox_dir = os.path.join(self.label_path, 'gt_center', f"{file_name}.txt")
        axis_dir = os.path.join(self.label_path, 'gt_logi', f"{file_name}.txt")
        table_units = TableEval.load_tabu(bbox_dir, axis_dir)
        table = TableWtw(TableEval.bubble_sort(table_units))
        return table

    def load_one_label_from_coco(self, image_id) -> TableWtw:
        ann_ids = self.coco_data.getAnnIds(imgIds=[image_id])
        anns = self.coco_data.loadAnns(ids=ann_ids)

        table_units = []
        for ann in anns:
            bbox = ann['segmentation'][0]
            logi = ann['logic_axis'][0]

            table_unit = TableUnit(bbox, logi)
            table_units.append(table_unit)

        table = TableWtw(TableEval.bubble_sort(table_units))
        return table

    def get_collate_fn(self, ):
        collate_fn = DataCollatorTableWtw()
        return collate_fn

    def get_image_path_from_meta(self, meta):
        image_id = [meta.cpu().numpy()[0][7]]
        image_name = self.get_image_path_from_id(image_id, )
        return image_name

    def get_image_path_from_id(self, image_id):
        single = False
        if not isinstance(image_id, list):
            image_id = [image_id]
            single = True
        file_paths = []
        for img_id in image_id:
            file_name = self.img_id_mapping[img_id]
            file_path = os.path.join(self.image_path, file_name)
            file_paths.append(file_path)

        if single:
            file_paths = file_paths[0]

        return file_paths

    def get_label_old(self, idx, one_item):
        if self.load_all:
            label = self.labels[idx]
        else:
            if self.coco_image_id is not None:
                label = self.load_one_label_from_coco(self.coco_image_id[idx])
            else:
                label = self.load_one_label_from_txt(self.images[idx])
        label_json = label.to_json()
        one_item["label"] = label_json

    def __len__(self, ):
        return len(self.images)

    def __getitem__(self, idx):
        image = os.path.join(self.image_path, self.images[idx])

        if self.labels is None:
            one_item = self.pre_processor(image)[0]
        else:
            image_id = self.coco_image_id[idx]
            ann_ids = self.coco_data.getAnnIds(imgIds=[image_id])
            anns = self.coco_data.loadAnns(ids=ann_ids)

            if self.split == "train":
                one_item = self.label_utils.get_label_from_coco(image_path=image,
                                                                anns=anns,
                                                                image_id=image_id)
            else:
                one_item = self.label_utils.get_image_eval(image_path=image, image_id=image_id)
        one_item = TableLorePreProcessor.update_meta(one_item)
        return one_item


@dataclass
class DataCollatorTableWtw:

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch_images = []
        batch_metas = []
        batch_inputs = []
        batch_label = []

        for feature in features:
            # print(f"feature: {feature.keys()}")
            batch_images.append(feature["image"])
            batch_metas.append(feature["meta"])
            batch_inputs.append(feature["inputs"])
            if "label" in feature:
                batch_label.append(feature["label"])

        # padding
        batch_images = torch.concat(batch_images)
        batch_metas = torch.concat(batch_metas)

        batch = {
            "image": batch_images,
            "meta": batch_metas,
            "inputs": batch_inputs,
        }

        if len(batch_label) > 0:
            batch["label"] = batch_label

        return batch
