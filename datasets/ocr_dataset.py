# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from pathlib import Path

import os
import random
import torch
import torch.utils.data
from torch.utils.data import ConcatDataset
import numpy as np
import datasets.transforms as T

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import csv
import bezier
import cv2


def adjust_pts(pts):
    x_pos_sort_idx = np.argsort(
        np.array([pts[0], pts[2], pts[4], pts[6]]))
    y_pos_sort_idx = np.argsort(
        np.array([pts[1], pts[3], pts[5], pts[7]]))

    pts_dict = {}

    first_pt_idx = -1
    second_pt_idx = -1
    third_pt_idx = -1
    fourth_pt_idx = -1
    if pts[2 * x_pos_sort_idx[0]] == pts[2 * x_pos_sort_idx[1]]:
        if pts[2 * x_pos_sort_idx[0] + 1] < pts[2 * x_pos_sort_idx[1] + 1]:
            first_pt_idx = x_pos_sort_idx[0]
        else:
            first_pt_idx = x_pos_sort_idx[1]
    else:
        first_pt_idx = x_pos_sort_idx[0]
    pts_dict[first_pt_idx] = 1

    slope_vals = []
    slope_idx = []
    for ptidx in range(0, 4):
        if ptidx in pts_dict:
            continue

        slopeval = 1.0 * (pts[2 * ptidx + 1] - pts[2 * first_pt_idx + 1]
                          ) / (1e-6 + pts[2 * ptidx] - pts[2 * first_pt_idx])
        slope_vals.append(slopeval)
        slope_idx.append(ptidx)

    slope_sort_idx = np.argsort(np.array(slope_vals))
    third_pt_idx = slope_idx[slope_sort_idx[1]]
    pts_dict[third_pt_idx] = 3

    a = (pts[2 * third_pt_idx + 1] - pts[2 * first_pt_idx + 1]) * \
        1.0 / (pts[2 * third_pt_idx] - pts[2 * first_pt_idx] + 1e-6)
    b = (pts[2 * third_pt_idx] * pts[2 * first_pt_idx + 1] - pts[2 * first_pt_idx] *
         pts[2 * third_pt_idx + 1]) * 1.0 / (pts[2 * third_pt_idx] - pts[2 * first_pt_idx] + 1e-6)
    for ptidx in range(0, 4):
        if ptidx in pts_dict:
            continue
        if pts[2 * ptidx + 1] < a * pts[2 * ptidx] + b:
            second_pt_idx = ptidx

    for ptidx in range(0, 4):
        if ptidx in pts_dict:
            continue
        if pts[2 * ptidx + 1] > a * pts[2 * ptidx] + b:
            fourth_pt_idx = ptidx

    pts_dict[second_pt_idx] = 2
    pts_dict[fourth_pt_idx] = 4

    ptsRealIdx = [first_pt_idx, second_pt_idx, third_pt_idx, fourth_pt_idx]
    start_ptIdx1 = -1
    start_ptIdx2 = -1
    if pts[2 * first_pt_idx] < pts[2 * third_pt_idx]:
        start_ptIdx1 = first_pt_idx
    else:
        start_ptIdx1 = third_pt_idx
    if pts[2 * second_pt_idx] < pts[2 * fourth_pt_idx]:
        start_ptIdx2 = second_pt_idx
    else:
        start_ptIdx2 = fourth_pt_idx

    start_ptIdx = -1
    if pts[2 * start_ptIdx1 + 1] < pts[2 * start_ptIdx2 + 1]:
        start_ptIdx = start_ptIdx1
    else:
        start_ptIdx = start_ptIdx2

    if start_ptIdx == second_pt_idx:
        ptsRealIdx = [second_pt_idx, third_pt_idx, fourth_pt_idx, first_pt_idx]
    elif start_ptIdx == third_pt_idx:
        ptsRealIdx = [third_pt_idx, fourth_pt_idx, first_pt_idx, second_pt_idx]
    elif start_ptIdx == fourth_pt_idx:
        ptsRealIdx = [fourth_pt_idx, first_pt_idx, second_pt_idx, third_pt_idx]

    pts_final = [-1, -1, -1, -1, -1, -1, -1, -1]
    if first_pt_idx == -1 or second_pt_idx == -1 or third_pt_idx == -1 or fourth_pt_idx == -1:
        return pts_final

    pts_final[0] = pts[2 * ptsRealIdx[0]]
    pts_final[1] = pts[2 * ptsRealIdx[0] + 1]
    pts_final[2] = pts[2 * ptsRealIdx[1]]
    pts_final[3] = pts[2 * ptsRealIdx[1] + 1]
    pts_final[4] = pts[2 * ptsRealIdx[2]]
    pts_final[5] = pts[2 * ptsRealIdx[2] + 1]
    pts_final[6] = pts[2 * ptsRealIdx[3]]
    pts_final[7] = pts[2 * ptsRealIdx[3] + 1]

    return pts_final

class CocoDetection():
    def __init__(self, args, image_set, img_folder, ann_file, transforms, return_masks, dataset_name, max_length, point_type, word_len, batch_aug):
        self.dataset_name = dataset_name
        self.image_folder = img_folder
        self._transforms = transforms
        self.point_type = point_type
        self.word_len = word_len
        self.batch_aug = batch_aug
        self.letters = args.chars
        self.annotations = []
        self.point_type = point_type
        self.num_bins = args.bins
        self.csv_file = ann_file
        self.chars = args.chars
        self.padding_index = args.pad_rec_index
        self.category_start_index_text = args.category_start_index_text
        self.no_known_char = args.no_known_char
        self.image_set = image_set
        self.train_stage = args.train_stage
        self.test_type = args.test_type

        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)  
            for row in reader:
                self.annotations.append(row)

    def tokenize(self, text):
        text = text[0]
        seq = []
        for i in range(len(text)):
            if text[i] in self.chars:
                seq.append(int(self.chars.index(text[i]))+self.category_start_index_text)
            else:
                seq.append(int(self.no_known_char))
        if len(seq)>self.word_len:
            seq = seq[:self.word_len]
        else:
            n = self.word_len-len(seq)
            seq.extend([self.padding_index] * n)
        return seq

    def tokenize_vqa(self, text):
        text = text[0]
        seq = []
        for i in range(len(text)):
            if text[i] in self.chars:
                seq.append(int(self.chars.index(text[i]))+self.category_start_index_text)
            else:
                seq.append(int(self.no_known_char))
        if len(seq)>256:
            seq = seq[:256]
        else:
            n = 256-len(seq)
            seq.extend([self.padding_index] * n)
        return seq

    def get_prompt(self,anno,dataset_name,train_stage):
        if not anno and self.image_set == "train": 
            all_prompts = ["all text","a photo of text"]
            concate_prompat = str()
            for prompt in all_prompts:
                concate_prompat+="<"+prompt+">"
            return [], [], concate_prompat
        if not anno and self.image_set != "train": 
            all_prompts = ["all text","a photo of text"]
            concate_prompat = str()
            for prompt in all_prompts:
                concate_prompat+="<"+prompt+">"
            return [], [], concate_prompat

        num_count = 0
        len_list = []
        start_character = []
        end_character = []
        all_character = set()
        for ele in anno:
            text = ele['rec']
            if text[0]:
                if text[0].isdigit():
                    num_count += 1
                len_list.append(len(text[0]))
                start_character.append(text[0][0])
                end_character.append(text[0][-1])
                all_character.update(list(text[0]))
        min_len = min(len_list)
        max_len = max(len_list)
        random_len = random.choice(len_list)
        random_flag = 0
        if min_len == max_len:
            random_flag = 1
        if max_len - min_len >= 2:
            random_num1 = random.randint(min_len + 1, max_len - 1)
            random_num2 = random.randint(min_len + 1, max_len - 1)
        else:
            random_num1 = min_len
            random_num2 = max_len

        prompts1 = ["all text",
        "text of more than {} characters".format(random_num1),
        "text of less than {} characters".format(random_num2),
        "text of",
        "text starts with the character",
        "text ends with the character",
        "text contains the character",
        "text of {} characters".format(random_len)
        ]

        if len(len_list)<2 or random_flag == 1 or train_stage == "pretrain":
            prompts1_prob = [1,0,0,0,0,0,0,0]
        else:
            prompts1_prob = [0.3,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

        if num_count > 0:
            prompts1.append("text of number")
            prompts1.append("text of char")
            prompts1_prob = [0.25,0.05,0.05,0.05,0.2,0.2,0.05,0.05,0.05,0.05]
            if train_stage == "pretrain":
                prompts1_prob = [1,0,0,0,0,0,0,0,0,0]

        prompt = np.random.choice(prompts1, p=prompts1_prob)
        points = [ele['segmentation'][0] for ele in anno]  
        text = [ele['rec'] for ele in anno]  
        filtered_points = []
        filtered_text = []
        if self.image_set == "train":
            if prompt == prompts1[3]:
                random_words = random.sample(text, random.randint(1, min(10, len(text))))
                random_words_str = ','.join([''.join(words) for words in random_words])
            elif prompt == prompts1[4]:
                random_char = random.choice(start_character)
            elif prompt == prompts1[5]:
                random_char = random.choice(end_character)
            elif prompt == prompts1[6]:
                random_char = random.choice(list(all_character))
            for p, t in zip(points, text):
                if prompt == prompts1[0]:
                    filtered_points.append(p)
                    filtered_text.append(t)
                elif "text of number" in prompts1 and prompt == "text of number" and t[0].isdigit():
                    filtered_points.append(p)
                    filtered_text.append(t)
                elif "text of char" in prompts1 and prompt == "text of char" and not t[0].isdigit():
                    filtered_points.append(p)
                    filtered_text.append(t)
                elif prompt == prompts1[1] and len(t[0]) > random_num1:
                    filtered_points.append(p)
                    filtered_text.append(t)
                elif prompt == prompts1[2] and len(t[0]) < random_num2:
                    filtered_points.append(p)
                    filtered_text.append(t)
                elif prompt == prompts1[3] and t in random_words:
                    filtered_points.append(p)
                    filtered_text.append(t)
                elif prompt == prompts1[4] and t[0].startswith(random_char):
                    filtered_points.append(p)
                    filtered_text.append(t)
                elif prompt == prompts1[5] and t[0].endswith(random_char):
                    filtered_points.append(p)
                    filtered_text.append(t)
                elif prompt == prompts1[6] and random_char in t[0]:
                    filtered_points.append(p)
                    filtered_text.append(t)
                elif prompt == prompts1[7] and len(t[0]) == random_len:
                    filtered_points.append(p)
                    filtered_text.append(t)
            if prompt == prompts1[3]:
                prompt = "text of {}".format(random_words_str)
            elif prompt == prompts1[4]:
                prompt = "text starts with the character '{}'".format(random_char)
            elif prompt == prompts1[5]:
                prompt = "text ends with the character '{}'".format(random_char)
            elif prompt == prompts1[6]:
                prompt = "text contains the character '{}'".format(random_char)
            
            all_prompts = []
            all_prompts.append(prompt)
            caption_text = "a photo of text"
            all_prompts.append(caption_text)
        else:
            filtered_points = points
            filtered_text = text
            all_prompts = ["all text","a photo of text"]
        concate_prompat = str()
        for prompt in all_prompts:
            concate_prompat+="<"+prompt+">"
        return filtered_points, filtered_text, concate_prompat

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        annotation = self.annotations[index]
        image_name = annotation[0]
        image_path = os.path.join(self.image_folder,image_name)
        image = Image.open(image_path).convert("RGB")   
        image_w, image_h = image.size
        anno = []
        if self.train_stage == "sft" and self.image_set == "train":
            text_annotations = annotation[1].split('&&tab&&')
            question = text_annotations[0]
            answers = text_annotations[1].split('&rec&')
            answers = answers[0]
            recog = self.tokenize_vqa([answers])
            recog = torch.tensor(recog, dtype=torch.long)
            target = {}
            target['dataset_name'] = self.dataset_name
            target['image_id'] = torch.tensor([index])
            target['file_name'] = image_name
            target['image_folder'] = self.image_folder
            image_size = torch.tensor([int(image_h), int(image_w)])
            target['orig_size'] = image_size
            target['size'] = image_size
            target['recog'] = recog
            target['prompts'] = question
            target['point_type'] = None
            image, target = self._transforms(image, target)
            return image, target
        if self.test_type == "vqa" and self.image_set == "val":
            if len(annotation[1]) != 0:
                question = annotation[1]
            if len(annotation) > 2:
                q_id = annotation[2]
            target = {}
            target['dataset_name'] = self.dataset_name
            target['image_id'] = torch.tensor([index])
            target['file_name'] = image_name
            target['image_folder'] = self.image_folder
            image_size = torch.tensor([int(image_h), int(image_w)])
            target['orig_size'] = image_size
            target['size'] = image_size
            target['prompts'] = question
            target['point_type'] = None
            target['recog'] = []
            target['question_id'] = q_id
            image, target = self._transforms(image, target)
            return image, target
        if len(annotation[1]) != 0:
            text_annotations = annotation[1].split('&&tab&&')
            for text_annotation in text_annotations:
                all_values = text_annotation.split('&rec&')
                points = all_values[0].split(",")
                coords = [int(float(x)) for x in points]
                if len(coords)<8:
                    continue
                text = all_values[1]
                if text=="###":
                    continue
                anno.append({'segmentation': [coords], 'rec': [text]})

        target = {}
        target['dataset_name'] = self.dataset_name
        target['image_id'] = torch.tensor([index])
        target['file_name'] = image_name
        target['image_folder'] = self.image_folder
        image_size = torch.tensor([int(image_h), int(image_w)])
        target['orig_size'] = image_size 
        target['size'] = image_size 

        filtered_points, filtered_text, all_prompts = self.get_prompt(anno,self.dataset_name,self.train_stage)

        recog = [self.tokenize(ele) for ele in filtered_text]
        recog = torch.tensor(recog, dtype=torch.long).reshape(-1, self.word_len)
        target['recog'] = recog

        target['prompts'] = all_prompts

        seg_pts = filtered_points
        if self.point_type=="bezier":
            polygons = []
            for bezier_points in seg_pts:
                polygon = _bezier_to_poly(bezier_points)
                polygons.append(polygon.flatten().tolist())
            seg_pts = polygons
            seg_pts = convert_to_min_area_rect(seg_pts)
            bboxes = [[min(coords[0::2]), min(coords[1::2]), max(coords[0::2]), max(coords[1::2])] for coords in seg_pts]
            seg_pts = torch.tensor(seg_pts, dtype=torch.float32).reshape(-1, 8)
        else:
            seg_pts = convert_to_min_area_rect(seg_pts)
            bboxes = [[min(coords[0::2]), min(coords[1::2]), max(coords[0::2]), max(coords[1::2])] for coords in seg_pts]
            seg_pts = torch.tensor(seg_pts, dtype=torch.float32).reshape(-1, 8)
        bboxes = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        target['seg_pts'] = seg_pts
        target['point_type'] = "quad"
        target['bboxes'] = bboxes

        image, target = self._transforms(image, target)
        return image, target

def sample_bezier_curve(bezier_pts, num_points=10, mid_point=False):
    curve = bezier.Curve.from_nodes(bezier_pts.transpose())
    if mid_point:
        x_vals = np.array([0.5])
    else:
        x_vals = np.linspace(0, 1, num_points)
    points = curve.evaluate_multi(x_vals).transpose()
    return points 

def bezier2bbox(bezier_pts):
    bezier_pts = bezier_pts.reshape(8, 2)
    points1 = sample_bezier_curve(bezier_pts[:4], 20)
    points2 = sample_bezier_curve(bezier_pts[4:], 20)
    points = np.concatenate((points1, points2))
    xmin = np.min(points[:, 0])
    ymin = np.min(points[:, 1])
    xmax = np.max(points[:, 0])
    ymax = np.max(points[:, 1])
    return [xmin, ymin, xmax, ymax]

def convert_to_min_area_rect(coords_list):
    result = []
    for coords in coords_list:
        if len(coords) > 8:
            np_coords = np.array(coords).reshape(-1, 2)
            rect = cv2.minAreaRect(np_coords)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            adjust_pts
            coords_order = adjust_pts(box.flatten().tolist())
            result.append(coords_order)
        else:
            coords = adjust_pts(coords)
            result.append(coords)
    return result

def _bezier_to_poly(bezier):
    bezier = np.array(bezier)
    u = np.linspace(0, 1, 8)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
        + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
        + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
        + np.outer(u ** 3, bezier[:, 3])
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)
    points = np.round(points).astype(int)
    return points

def make_transforms(image_set, max_size_train, min_size_train, max_size_test, min_size_test,
                         crop_min_ratio, crop_max_ratio, crop_prob, rotate_max_angle, rotate_prob,
                         brightness, contrast, saturation, hue, distortion_prob, gaussian_blur_prob=0.3):

    transforms = []
    if image_set == 'train':
        transforms.append(T.RandomSizeCrop(crop_min_ratio, crop_max_ratio, crop_prob))
        transforms.append(T.RandomRotate(rotate_max_angle, rotate_prob))
        transforms.append(T.RandomResize(min_size_train, max_size_train))
        transforms.append(T.RandomDistortion(brightness, contrast, saturation, hue, distortion_prob))
        transforms.append(T.RandomGaussianBlur(gaussian_blur_prob))
    if image_set == 'val':
        transforms.append(T.RandomResize([min_size_test], max_size_test))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(None, None))

    return T.Compose(transforms)



def build(image_set, args):
    root = Path(args.data_root)
    if image_set == 'train':
        dataset_names = args.train_dataset.split(':')
    elif image_set == 'val':
        dataset_names = args.val_dataset.split(':')
    
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name == 'totaltext_train':
            img_folder = root / "totaltext" / "train_images"; ann_file = root / "totaltext" / "totaltext_train.csv"; point_type = "bezier"
        elif dataset_name == 'totaltext_val':
            img_folder = root / "totaltext" / "test_images"; ann_file = root / "totaltext" / "totaltext_test.csv"; point_type = "bezier"
        elif dataset_name == 'ic13_train':
            img_folder = root / "ic13" / "train_images"; ann_file = root / "ic13" / "ic13_train.csv"; point_type = "quad"
        elif dataset_name == 'ic13_val':
            img_folder = root / "ic13" / "test_images"; ann_file = root / "ic13" / "ic13_test.csv"; point_type = "quad"
        elif dataset_name == 'ic15_train':
            img_folder = root / "ic15" / "train_images"; ann_file = root / "ic15" / "ic15_train.csv"; point_type = "quad"
        elif dataset_name == 'ic15_val':
            img_folder = root / "ic15" / "test_images"; ann_file = root / "ic15" / "ic15_test.csv"; point_type = "quad"
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
        transforms = make_transforms(image_set, args.max_size_train, args.min_size_train,
              args.max_size_test, args.min_size_test, args.crop_min_ratio, args.crop_max_ratio,
              args.crop_prob, args.rotate_max_angle, args.rotate_prob, args.brightness, args.contrast,
              args.saturation, args.hue, args.distortion_prob)
        dataset = CocoDetection(args, image_set, img_folder, ann_file, transforms=transforms, return_masks=args.masks, dataset_name=dataset_name, max_length=args.max_length, point_type = point_type, word_len=25,  batch_aug=True)
        datasets.append(dataset)
    
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)

    return dataset
