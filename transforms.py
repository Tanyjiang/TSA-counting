from PIL import Image
import numpy as np
import torch
import cv2
import random
from torchvision.transforms import functional
import torchvision.transforms.functional as F

def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(inner_down - inner_up, 0.0)
    return inner_area

def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map


    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h - 1] * num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w - 1] * num_gt).astype(int))
    p_index = torch.from_numpy(p_h * im_width + p_w).to(torch.int64)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index,src=torch.ones(im_width * im_height)).view(im_height, im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map

class Transforms(object):
    def __init__(self, scale, crop, stride, gamma, dataset, d_ratio):
        self.scale = scale
        self.crop = crop
        self.stride = stride
        self.gamma = gamma
        self.dataset = dataset
        self.d_ratio = d_ratio

    def __call__(self, image, keypoints, attention):
        # random resize
        height, width = image.size[1], image.size[0]
        if self.dataset == 'train':
            if height < width:
                short = height
            else:
                short = width
            if short < 320:
                scale = 320 / short
                height = round(height * scale)
                width = round(width * scale)
                image = image.resize((width, height), Image.BILINEAR)
                if len(keypoints) > 0:
                    keypoints = keypoints * scale



        h, w = 585, 1024
        dh = 0
        dw = 0
        image = image.crop((dw, dh, dw + w, dh + h))

        if len(keypoints) > 0:
            ker_1 = keypoints
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(dw, dh, dw + w, dh + h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)
            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [dw, dh]
        else:
            target = np.array([])
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                image = F.hflip(image)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                image = F.hflip(image)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image, torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(gt_discrete.copy()).float()
