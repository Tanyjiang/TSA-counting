from iCaRL import iCaRLmodel
import torch
from clip import clip
import time

numclass = 1
batch_size = 16
downsample_ratio = 8
crop_size = 224
task_size = 1
memory_size = 150
epochs= 50
learning_rate = 5e-4
dataset_dir = '/path/to/your/dataset'
dataset_config_path = 'dataset_config.json'

test_phase = None

start_time_total = time.time()
feature_extractor, _ = clip.load('ViT-B/16', torch.device('cuda'))
model = iCaRLmodel(numclass, feature_extractor, batch_size, task_size, memory_size,
                   epochs, learning_rate, crop_size, downsample_ratio, dataset_dir,
                   dataset_config_path, test_phase)


for i in range(1):
    model.afterTrain()

end_time_total = time.time()
print('total time cost:{:.2f} hours'.format((end_time_total-start_time_total)/3600))
