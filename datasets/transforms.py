# ------------------------------------------------------------------------
# Copyright (2023) Bytedance Ltd. and/or its affiliates
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import cv2
import torch
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from copy import deepcopy
import bezier
import math
from PIL import ImageFilter

class RandomGaussianBlur(object):
    def __init__(self, prob=0.5, radius_min=0.1, radius_max=2):
        self.prob = prob
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img, target=None):
        if random.random() < self.prob:
            radius = random.uniform(self.radius_min, self.radius_max)
            img = img.filter(ImageFilter.GaussianBlur(radius))
            return img, target
        else:
            return img, target

def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = math.ceil(size * h / w)
        else:
            oh = size
            ow = math.ceil(size * w / h)

        if max_size is not None:
            ow = min(ow, max_size)
            oh = min(oh, max_size)

        return (int(oh), int(ow))

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "bboxes" in target:
        boxes = target["bboxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["bboxes"] = scaled_boxes

    if target['point_type'] == 'poly':
            target['seg_pts'] = target['seg_pts'] * torch.as_tensor([ratio_width, ratio_height] * 8)
    if target['point_type'] == 'quad':
            target['seg_pts'] = target['seg_pts'] * torch.as_tensor([ratio_width, ratio_height] * 4)

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


def pad(image, target, padding):
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target

class RandomSizeCrop(object):
    def __init__(self, min_size_ratio, max_size_ratio, prob):
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.prob = prob 

    def __call__(self, image, target):
        if random.random() > self.prob or len(target['bboxes']) == 0:
            return image, target

        for _ in range(100):
            crop_w = int(image.width * random.uniform(self.min_size_ratio, self.max_size_ratio))
            crop_h = int(image.height * random.uniform(self.min_size_ratio, self.max_size_ratio))
            crop_region = transforms.RandomCrop.get_params(image, [crop_h, crop_w])
            cropped_image, cropped_target= self.crop(deepcopy(image), deepcopy(target), crop_region)
            if not cropped_image is None:
                return cropped_image, cropped_target

        print('Can not be cropped with texts')
        return image, target
    
    def crop(self, image, target, crop_region):
        bboxes = target['bboxes']
        crop_region, keep_instance = self.adjust_crop_region(bboxes, crop_region)
        
        if crop_region is None:
            return None, None

        cropped_image = F.crop(image, *crop_region)

        rg_ymin, rg_xmin, rg_h, rg_w = crop_region
        target['size'] = torch.tensor([rg_h, rg_w])
        if bboxes.shape[0] > 0:
            target['bboxes'] = target['bboxes'] - torch.tensor([rg_xmin, rg_ymin] * 2)
            if target['point_type'] == 'poly':
                target['seg_pts'] = target['seg_pts'] - torch.tensor([rg_xmin, rg_ymin] * 8)
            else:
                target['seg_pts'] = target['seg_pts'] - torch.tensor([rg_xmin, rg_ymin] * 4)
            for k in ['recog', 'bboxes', 'seg_pts']:
                target[k] = target[k][keep_instance]

        return cropped_image, target

    def adjust_crop_region(self, bboxes, crop_region):
        rg_ymin, rg_xmin, rg_h, rg_w = crop_region 
        rg_xmax = rg_xmin + rg_w 
        rg_ymax = rg_ymin + rg_h 

        pre_keep = torch.zeros((bboxes.shape[0], ), dtype=torch.bool)
        while True:
            ov_xmin = torch.clamp(bboxes[:, 0], min=rg_xmin)
            ov_ymin = torch.clamp(bboxes[:, 1], min=rg_ymin)
            ov_xmax = torch.clamp(bboxes[:, 2], max=rg_xmax)
            ov_ymax = torch.clamp(bboxes[:, 3], max=rg_ymax)
            ov_h = ov_ymax - ov_ymin 
            ov_w = ov_xmax - ov_xmin 
            keep = torch.bitwise_and(ov_w > 0, ov_h > 0)

            if (keep == False).all():
                return None, None

            if keep.equal(pre_keep):
                break 

            keep_bboxes = bboxes[keep]
            keep_bboxes_xmin = int(min(keep_bboxes[:, 0]).item())
            keep_bboxes_ymin = int(min(keep_bboxes[:, 1]).item())
            keep_bboxes_xmax = int(max(keep_bboxes[:, 2]).item())
            keep_bboxes_ymax = int(max(keep_bboxes[:, 3]).item())
            rg_xmin = min(rg_xmin, keep_bboxes_xmin)
            rg_ymin = min(rg_ymin, keep_bboxes_ymin)
            rg_xmax = max(rg_xmax, keep_bboxes_xmax)
            rg_ymax = max(rg_ymax, keep_bboxes_ymax)

            pre_keep = keep
        
        crop_region = (rg_ymin, rg_xmin, rg_ymax - rg_ymin, rg_xmax - rg_xmin)
        return crop_region, keep

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target

class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):

        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        
        if target['point_type'] == 'poly':
            target['seg_pts'] = target['seg_pts'] / torch.tensor([w, h] * 8, dtype=torch.float32)
        elif  target['point_type'] == 'quad':
            target['seg_pts'] = target['seg_pts'] / torch.tensor([w, h] * 4, dtype=torch.float32)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

def read_word_list(word_list_file):
    words = open(word_list_file, 'r').read().splitlines()
    words = words[1:]
    words = [word.split('\t')[0] for word in words]
    words = [word.split('/')[0] for word in words]
    return words

class RandomRotate(object):
    def __init__(self, max_angle, prob):
        self.max_angle = max_angle 
        self.prob = prob 

    def __call__(self, image, target):
        if random.random() > self.prob or target['bboxes'].shape[0] <= 0:
            return image, target
        seed_rot = random.random()
        if seed_rot>0.9:
            angle = 90
        elif 0.9>=seed_rot>0.8:
            angle = -90
        elif 0.8 >= seed_rot > 0.70:
            angle = 180
        else:
            angle = random.uniform(-self.max_angle, self.max_angle)
        image_w, image_h = image.size
        rotation_matrix = cv2.getRotationMatrix2D((image_w//2, image_h//2), angle, 1)
        
        image = image.rotate(angle, expand=True)

        new_w, new_h = image.size 
        target['size'] = torch.tensor([new_h, new_w])
        pad_w = (new_w - image_w) / 2
        pad_h = (new_h - image_h) / 2

        seg_pts = target['seg_pts'].numpy()
        if target['point_type'] == 'poly':
            seg_pts = seg_pts.reshape(-1, 8, 2)
            seg_pts = self.rotate_points(seg_pts, rotation_matrix, (pad_w, pad_h))
            seg_pts = seg_pts.reshape(-1, 16)
        elif target['point_type'] == 'quad':
            seg_pts = seg_pts.reshape(-1, 4, 2)
            seg_pts = self.rotate_points(seg_pts, rotation_matrix, (pad_w, pad_h))
            seg_pts = seg_pts.reshape(-1, 8)
        target['seg_pts'] = torch.from_numpy(seg_pts).type(torch.float32)
        bboxes = [cv2.boundingRect(ele.astype(np.uint).reshape((-1, 1, 2))) for ele in seg_pts]
        


        target['bboxes'] = torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4)
        target['bboxes'][:, 2] += target['bboxes'][:, 0]
        target['bboxes'][:, 3] += target['bboxes'][:, 1]

        return image, target

    def rotate_points(self, coords, rotation_matrix, paddings):
        coords = np.pad(coords, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=1)
        coords = np.dot(coords, rotation_matrix.transpose())
        coords[:, :, 0] += paddings[0]
        coords[:, :, 1] += paddings[1]
        return coords

class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target

def sample_bezier_curve(bezier_pts, num_points=10, mid_point=False):
    curve = bezier.Curve.from_nodes(bezier_pts.transpose())
    if mid_point:
        x_vals = np.array([0.5])
    else:
        x_vals = np.linspace(0, 1, num_points)
    points = curve.evaluate_multi(x_vals).transpose()
    return points
