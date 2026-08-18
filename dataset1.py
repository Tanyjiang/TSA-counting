import cv2
from torch.utils import data
from PIL import Image
import numpy as np
import random
import torch
from torchvision import transforms
import os
import glob
from torchvision.transforms import functional
import json
from clip import tokenize

class Dataset1(data.Dataset):
    def __init__(self, dataset, increntmal_phase, exemplar_set, exemplar_set_gt,
                 crop_size, downsample_ratio, dataset_dir, dataset_config_path):
        self.dataset = dataset
        self.crop_size = crop_size
        self.d_ratio = downsample_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.label_list = []
        self.image_list = []
        self.target = []
        self.eval_groups = []
        self.increntmal_phase = increntmal_phase

        dataset_config_path = os.path.abspath(dataset_config_path)
        with open(dataset_config_path, encoding='utf-8') as f:
            dataset_config = json.load(f)

        self.stages = dataset_config['stages']
        if not self.stages:
            raise ValueError('dataset_config.json must contain at least one stage.')
        if not 0 <= increntmal_phase < len(self.stages):
            raise ValueError(
                'Incremental phase {} is outside the configured range 0-{}.'.format(
                    increntmal_phase, len(self.stages) - 1
                )
            )

        self.background_class = dataset_config.get('background_class', 'background')
        self.class_name = [self.background_class] + [stage['class_name'] for stage in self.stages]
        self.class_prefixes = [stage.get('file_prefix', stage['class_name']) for stage in self.stages]

        class_prompt = dataset_config.get('prompt_file', 'prompt/class_prompt.json')
        if not os.path.isabs(class_prompt):
            class_prompt = os.path.join(os.path.dirname(dataset_config_path), class_prompt)
        with open(class_prompt, encoding='utf-8') as f:
            self.class_prompt = json.load(f)

        for class_index, class_key in enumerate(self.class_name):
            if class_key not in self.class_prompt:
                raise KeyError("Missing prompt for class '{}' in {}.".format(class_key, class_prompt))
            self.class_prompt.setdefault(str(class_index), self.class_prompt[class_key])

        split_directory = dataset + '_data'
        if self.dataset in ('test', 'val'):
            # Evaluate the newly introduced class first, followed by all old classes.
            eval_stage_indices = [increntmal_phase] + list(range(increntmal_phase))
            for stage_index in eval_stage_indices:
                stage = self.stages[stage_index]
                dataset_path = os.path.join(
                    dataset_dir, stage['directory'], split_directory, 'images'
                )
                stage_images = sorted(glob.glob(os.path.join(dataset_path, '*.jpg')))
                if not stage_images:
                    raise FileNotFoundError(
                        "No .jpg images found for class '{}' in {}.".format(
                            stage['class_name'], dataset_path
                        )
                    )

                start = len(self.image_list)
                self.image_list.extend(stage_images)
                self.label_list.extend([self._ground_truth_path(path) for path in stage_images])
                end = len(self.image_list)
                self.eval_groups.append({
                    'start': start,
                    'end': end,
                    'class_index': stage_index + 1,
                    'class_name': stage['class_name'],
                })

        elif self.dataset == 'train':
            stage = self.stages[increntmal_phase]
            dataset_path = os.path.join(
                dataset_dir, stage['directory'], split_directory, 'images'
            )
            stage_images = sorted(glob.glob(os.path.join(dataset_path, '*.jpg')))
            for image_path in stage_images:
                class_index = self._class_index_from_image(image_path)
                self.image_list.append(image_path)
                self.target.append(class_index)
                if class_index == 0:
                    self.label_list.append(np.empty((0, 3), dtype=np.float32))
                else:
                    self.label_list.append(self._ground_truth_path(image_path))

            if exemplar_set is not None:
                for set_index in range(len(exemplar_set)):
                    for exemplar_index in range(len(exemplar_set[set_index])):
                        image_path = exemplar_set[set_index][exemplar_index]
                        self.image_list.append(image_path)
                        self.label_list.append(exemplar_set_gt[set_index][exemplar_index])
                        self.target.append(self._class_index_from_image(image_path))
        else:
            raise ValueError("dataset must be one of: 'train', 'val', or 'test'.")

    @staticmethod
    def _ground_truth_path(image_path):
        split_path = os.path.dirname(os.path.dirname(image_path))
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(split_path, 'ground_truth', image_stem + '.npy')

    def _class_index_from_image(self, image_path):
        image_name = os.path.basename(image_path)
        for class_index, prefix in enumerate(self.class_prefixes, start=1):
            if image_name.startswith(prefix):
                return class_index
        return 0

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')

        if self.dataset == 'train':
            target = self.target[index]
            cls_name = self.class_name[target]
            text = self.class_prompt[cls_name]
            tokenized_text = []
            for i in range(len(text)):
                tokenized_prompt = tokenize(text[i])
                tokenized_text.append(tokenized_prompt)

            if target > 0:
                keypoints = np.load(self.label_list[index])
            else:
                keypoints = np.empty((0, 3), dtype=np.float32)
        else:
            keypoints = np.load(self.label_list[index])
            gt = len(keypoints)

        if self.dataset == 'train':
            image, points, targets, st_sizes = self.train_transform(image, keypoints)
            return image, points, targets, st_sizes, target, tokenized_text

        else:
            ht, wd = image.size[1], image.size[0]
            st_size = 1.0 * min(wd, ht)
            if st_size < self.crop_size:
                rr = 1.0 * self.crop_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                image = image.resize((wd, ht), Image.BILINEAR)
            assert st_size >= self.crop_size, print(wd, ht)

            image_224 = image.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            image_224 = functional.to_tensor(image_224)
            image_224 = functional.normalize(image_224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            image = functional.to_tensor(image)
            image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            image_name = os.path.splitext(os.path.basename(self.image_list[index]))[0]
            return image, gt, image_224, self.class_prompt, image_name

    def train_transform(self, img, keypoints):
        height, width = img.size[1], img.size[0]
        if height < width:
            short = height
        else:
            short = width

        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        if st_size < self.crop_size:
            rr = 1.0 * self.crop_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.crop_size, print(wd, ht)
        assert len(keypoints) >= 0

        height, width = img.size[1], img.size[0]
        h, w = self.crop_size, self.crop_size
        dh = random.randint(0, height - h)
        dw = random.randint(0, width - w)
        img = img.crop((dw, dh, dw + w, dh + h))

        if len(keypoints) > 0:
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = self.cal_innner_area(dw, dh, dw + w, dh + h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)
            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [dw, dh]
        else:
            target = np.array([])


        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if len(keypoints) > 0:
                keypoints[:, 0] = w - keypoints[:, 0]

        # random gamma
        if random.random() < 0.3:
            gamma = random.uniform(0.5, 1.5)
            img = functional.adjust_gamma(img, gamma)

        # random to gray
        if self.dataset == 'train':
            if random.random() < 0.1:
                img = functional.to_grayscale(img, num_output_channels=3)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            target.copy()).float(), short

    def cal_innner_area(self, c_left, c_up, c_right, c_down, bbox):
        inner_left = np.maximum(c_left, bbox[:, 0])
        inner_up = np.maximum(c_up, bbox[:, 1])
        inner_right = np.minimum(c_right, bbox[:, 2])
        inner_down = np.minimum(c_down, bbox[:, 3])
        inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(inner_down - inner_up, 0.0)
        return inner_area

    def __len__(self):
        return len(self.image_list)
