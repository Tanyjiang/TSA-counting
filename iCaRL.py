import glob
import json
from torch import Tensor
from typing import Optional, List
import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from losses.ot_loss import OT_Loss
from myNetwork import network
from utils.pytorch_utils import Save_Handle, AverageMeter
from torch.utils.data import DataLoader
from torchvision.transforms import functional
from datetime import datetime
import utils.log_utils as log_utils
import wandb
from  dataset1 import Dataset1
import os
import cv2
import time
from loss.bay_loss import Bay_Loss
from loss.post_prob import Post_Prob
import matplotlib.pyplot as plt
plt.switch_backend('agg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]
    targets = [[transposed_batch[2][i]] for i in range(len(transposed_batch[2]))]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    label = torch.tensor(transposed_batch[4])
    tokenized_text = transposed_batch[5]
    return images, points, targets, st_sizes, label, tokenized_text

def val_collate(batch):
    transposed_batch = list(zip(*batch))
    transposed_batch[0] = nested_tensor_from_tensor_list(transposed_batch[0])
    transposed_batch[2] = nested_tensor_from_tensor_list(transposed_batch[2])
    tokenized_text = transposed_batch[3]
    return transposed_batch[0], torch.tensor(batch[0][1]).unsqueeze(0), transposed_batch[2], tokenized_text, transposed_batch[4]

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:

        # TODO make it support different-sized images
        max_size = _max_by_axis_pad([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        for img, pad_img in zip(tensor_list, tensor):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    else:
        raise ValueError('not supported')
    return tensor

def _max_by_axis_pad(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)

    block = 224

    for i in range(2):
        maxes[i+1] = ((maxes[i+1] - 1) // block + 1) * block
    return maxes


def w_one_hot(labels, num_classes=4):
    """
    将标签转换为one-hot编码
    Args:
        labels: 原始标签，如 [0,1,2,3,4]
        num_classes: one-hot向量的维度，这里为4
    Returns:
        one-hot编码的张量
    """
    adjusted_labels = labels - 1
    mask = adjusted_labels >= 0
    one_hot = torch.zeros(len(labels), num_classes)
    valid_indices = torch.arange(len(labels))[mask]
    valid_labels = adjusted_labels[mask]

    one_hot[valid_indices, valid_labels] = 1

    return one_hot

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class iCaRLmodel:

    def __init__(self, numclass, feature_extractor, batch_size,
                 task_size, memory_size, epochs, learning_rate, crop_size, downsample_ratio,
                 dataset_dir, dataset_config_path, test_phase=None):

        super(iCaRLmodel, self).__init__()
        self.dataset_config_path = os.path.abspath(dataset_config_path)
        with open(self.dataset_config_path, encoding='utf-8') as f:
            dataset_config = json.load(f)
        self.stages = dataset_config['stages']
        if not self.stages:
            raise ValueError('dataset_config.json must contain at least one stage.')
        if len(self.stages) > 6:
            raise ValueError(
                'This CLIP adapter implementation supports at most 6 incremental stages.'
            )
        self.class_name = [dataset_config.get('background_class', 'background')] + [
            stage['class_name'] for stage in self.stages
        ]
        self.test_phase = len(self.stages) - 1 if test_phase is None else test_phase
        if not 0 <= self.test_phase < len(self.stages):
            raise ValueError(
                'test_phase {} is outside the configured range 0-{}.'.format(
                    self.test_phase, len(self.stages) - 1
                )
            )

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.increntmal_phase = 0
        self.model = network(feature_extractor, flag=1, num_classes=len(self.class_name))
        self.exemplar_set = []
        self.exemplar_set_gt = []
        self.class_mean_set = []
        self.numclass = numclass
        self.old_model = network(feature_extractor, flag=1, num_classes=len(self.class_name))
        self.test_model = network(feature_extractor, flag=1, num_classes=len(self.class_name))
        self.train_list = list()
        self.val_list = list()
        self.batchsize = batch_size
        self.memory_size = memory_size
        self.task_size = task_size
        self.workers = 4
        self.train_loader = None
        self.test_loader = None
        self.train_dataset = []
        self.image_list = list()
        self.label_list = list()
        self.crop_size = crop_size
        self.downsample_ratio = downsample_ratio
        self.dataset_dir = dataset_dir
        self.post_prob = Post_Prob(8.0, 224, 8, 0.15, True, device)
        self.mae_loss = nn.L1Loss().to(device)
        self.mse_loss = nn.MSELoss(reduction='sum').to(device)
        self.criterion_bay = Bay_Loss(True, device)


        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        self.logger = log_utils.get_logger(
            os.path.join('./checkpoint', "train-{:s}.log".format(time_str))
        )

    def beforeTrain(self):

        self.model.eval()
        self.train_loader, self.val_loader = self._get_train_and_val_dataloader()
        if self.numclass > self.task_size:
           self.model.Incremental_learning_weight(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_val_dataloader(self):
        self.train_dataset = glob.glob(
            os.path.join(
                self.dataset_dir,
                self.stages[self.increntmal_phase]['directory'],
                'train_data/images',
                '*.jpg'
            )
        )
        train_dataset = Dataset1('train', self.increntmal_phase, self.exemplar_set, self.exemplar_set_gt,
                                 self.crop_size, self.downsample_ratio, self.dataset_dir,
                                 self.dataset_config_path)
        train_loader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, collate_fn=train_collate,
                                  drop_last=True)
        print('{0}th phase: the length of the train_dataset:{1}'.format(self.increntmal_phase, len(train_dataset)))

        val_dataset = Dataset1('val', self.increntmal_phase, None, None, self.crop_size, self.downsample_ratio,
                               self.dataset_dir, self.dataset_config_path)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=val_collate)
        print('{0}th phase: the length of the val_dataset:{1}'.format(self.increntmal_phase, len(val_dataset)))

        return train_loader, val_loader




    def _test(self, test_loader, flag, model, crop_size):
        model.eval()
        mae_final = 0
        mse_final = 0
        mae_total = 0
        mse_total = 0
        all_test_dataset_img = test_loader.dataset.image_list
        all_test_dataset_label = test_loader.dataset.label_list
        all_test_dataset_target = test_loader.dataset.target
        eval_groups = test_loader.dataset.eval_groups
        if not eval_groups:
            raise ValueError('The evaluation dataset does not contain any class groups.')

        for group in eval_groups:
            correct = 0
            mae = 0
            mse = 0

            start, end = group['start'], group['end']
            test_loader.dataset.image_list = all_test_dataset_img[start:end]
            test_loader.dataset.label_list = all_test_dataset_label[start:end]
            class_sample_count = end - start
            if class_sample_count == 0:
                raise ValueError("Class '{}' has no evaluation samples.".format(group['class_name']))

            for i, (inputs, count, inputs_224, tokenized_text, name) in enumerate(test_loader):
                model = model.to(device)
                inputs = inputs.to(device)
                inputs_224 = inputs_224.to(device)

                count = count.to(device)

                with torch.no_grad():

                    crop_imgs = []
                    positions = []
                    b, c, h, w = inputs.size()
                    rh, rw = crop_size, crop_size
                    for i in range(0, h, rh):
                        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                        for j in range(0, w, rw):
                            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                            crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                            positions.append((gis, gie, gjs, gje))

                    crop_imgs = torch.cat(crop_imgs, dim=0)

                    all_outputs = []
                    nz, bz = crop_imgs.size(0), b
                    for i in range(0, nz, bz):
                        gs, gt = i, min(nz, i + bz)
                        crop_output, _, _, _, _ = model(crop_imgs[gs:gt],tokenized_text, 1, self.increntmal_phase)
                        all_outputs.append(crop_output)

                    output_ex = torch.cat(all_outputs, dim=0)
                    output = torch.zeros(
                        b, output_ex.shape[1], h, w,
                        device=output_ex.device,
                        dtype=output_ex.dtype,
                    )
                    for idx, (gis, gie, gjs, gje) in enumerate(positions):
                        crop_output = output_ex[idx]
                        output[:, :, gis:gie, gjs:gje] = crop_output

                    _, cls, _, _, _ = model(inputs_224,tokenized_text, 1, self.increntmal_phase)

                for index in range(cls.shape[0]):
                    channel_num = torch.argmax(cls[index]).item()
                    if channel_num == group['class_index']:
                        correct += 1

                    output = output[:, channel_num:channel_num + 1, :, :]
                    count_error = output.sum() - count.sum()
                    absolute_error = torch.abs(count_error).item()
                    squared_error = (count_error ** 2).item()
                    mae += absolute_error
                    mse += squared_error
                    mae_total += absolute_error
                    mse_total += squared_error

            cls_accuracy = correct / class_sample_count
            mae = mae / class_sample_count
            mse = mse / class_sample_count
            mse = mse ** 0.5
            print(
                'class:%s, mae:%.3f, mse:%.3f, cls_accuracy: %.6f, num: %d' % (
                    group['class_name'], mae, mse, cls_accuracy, class_sample_count
                )
            )

            mae_final += mae
            mse_final += mse

        mae_final = mae_final / len(eval_groups)
        mse_final = mse_final / len(eval_groups)
        test_loader.dataset.image_list = all_test_dataset_img
        test_loader.dataset.label_list = all_test_dataset_label
        test_loader.dataset.target = all_test_dataset_target


        mae_total = mae_total / len(all_test_dataset_img)
        mse_total = mse_total / len(all_test_dataset_img)
        mse_total = mse_total ** 0.5
        print(' **  MAE : %.2f ' % (mae_total))
        print(' **  MSE : %.2f ' % (mse_total))

        return mae_final, mse_final


    def afterTrain(self):
        mae = 0
        self.model.eval()

        self.increntmal_phase = self.test_phase
        checkpoint_val = torch.load(os.path.join('./checkpoint', 'checkpoint_best_' + str(self.increntmal_phase) + '.pth'))
        self.model.load_state_dict(checkpoint_val['model'])

        print('*begin Test*')
        start_time = time.time()
        test_dataset = Dataset1(
            'test', self.increntmal_phase, None, None,
            self.crop_size, self.downsample_ratio, self.dataset_dir,
            self.dataset_config_path
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=val_collate)

        print('{0}th phase: the length of the test_dataset:{1}'.format(self.increntmal_phase, len(test_dataset)))
        mae, mse = self._test(test_loader,'test', self.model, self.crop_size)

        self.logger.info(
            "{}th phase Test cost time:{:.2f} sec".format(
                self.increntmal_phase, time.time() - start_time))



    def _construct_exemplar_set_herding(self, images, m):

        if self.increntmal_phase == 0:
            foreground_images = []
            background_images = []
            foreground_prefix = self.stages[0].get(
                'file_prefix', self.stages[0]['class_name']
            )
            for index in range(len(images)):
                image_name = os.path.basename(images[index])
                if image_name.startswith(foreground_prefix):
                    foreground_images.append(images[index])
                else:
                    background_images.append(images[index])

            if foreground_images:
                class_mean, features = self.compute_class_mean(foreground_images)
                self.find_exemplar_set_and_exemplar_set_gt_herding(
                    class_mean, features, foreground_images, m
                )
            if background_images:
                class_mean, features = self.compute_class_mean(background_images)
                self.find_exemplar_set_and_exemplar_set_gt_herding(
                    class_mean, features, background_images, m
                )
        else:
            class_mean, feature_extractor_output = self.compute_class_mean(images)
            self.find_exemplar_set_and_exemplar_set_gt_herding(class_mean,feature_extractor_output,images,m)



    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            self.exemplar_set_gt[index] = self.exemplar_set_gt[index][:m]


    def compute_class_mean(self, images):
        self.model.eval()

        img=[]
        for i in range(len(images)):
            img.append(images[i])

        x = self.transform_image(img[0]).unsqueeze(0)
        x = x.to(device)
        with torch.no_grad():
            _, _, feature, _, _ = self.model(x, None, 0, self.increntmal_phase)

        feature = F.normalize(feature).cpu().numpy()
        for index in range(1,len(img)):

            img[index] = self.transform_image(img[index]).unsqueeze(0)
            img[index]=img[index].to(device)
            with torch.no_grad():
                _, _, feature_this, _, _ = self.model(img[index], None, 0, self.increntmal_phase)

            feature_this = F.normalize(feature_this).cpu().numpy()
            feature = np.concatenate((feature_this, feature), axis=0)

        class_mean = np.mean(feature, axis=0)
        return class_mean, feature

    def find_exemplar_set_and_exemplar_set_gt_herding(self,class_mean,feature_extractor_output,images,m):
        exemplar = []
        exemplar_gt = []
        now_class_mean = np.zeros((1, 128))
        for i in range(m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])
            image_name = os.path.basename(images[index])
            is_foreground = any(
                image_name.startswith(stage.get('file_prefix', stage['class_name']))
                for stage in self.stages
            )

            if is_foreground:
                split_path = os.path.dirname(os.path.dirname(images[index]))
                image_stem = os.path.splitext(image_name)[0]
                gt_path = os.path.join(split_path, 'ground_truth', image_stem + '.npy')
                exemplar_gt.append(gt_path)
            else:
                exemplar_gt.append(np.empty((0, 3), dtype=np.float32))
        self.exemplar_set.append(exemplar)
        self.exemplar_set_gt.append(exemplar_gt)
        print("the size of exemplar :%s" % (str(len(exemplar))))


    def transform_image(self, image):
        image = Image.open(image).convert('RGB')
        image = image.resize((self.crop_size, self.crop_size), Image.BILINEAR)

        image = functional.to_tensor(image)
        image = functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image
