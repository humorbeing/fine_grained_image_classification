
# pytorch
import torch
from torch.autograd import Variable
# torchvision
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
# more
import numpy as np
import copy
import time
# me
from package import me

# Hyper-parameter
test_set_size = 0.3
batch_size = 10
#####################

since = time.time()
dataset_path = '/media/ray/SSD/workspace/python/dataset/fish/original'
gpu = torch.cuda.is_available()


dataset_loaders,test_loader,\
dataset_sizes, dataset_classes\
    = me.train_set_test_set(
    dataset_path,
    test_set_size,
    batch_size=1,
    random_seed=169,
    # is_validation_set=True
)
print(dataset_sizes)
for i in dataset_loaders:
    print(len(dataset_loaders[i]))
time_elapsed = time.time() - since
print(
    'Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60
    )
)









