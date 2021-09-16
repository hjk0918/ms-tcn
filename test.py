from batch_gen import BatchGenerator
from model import MultiStageModel
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random


vid_list_file = 'C:/github/casual_tcn/cholec80/train.txt'
mapping_file = 'C:/github/casual_tcn/cholec80/action_mapping.txt'
feature_dir = 'C:/github/casual_tcn/cholec80/train_dataset/video_feature@2020/'
gt_dir = 'C:/github/casual_tcn/cholec80/train_dataset/annotation_folder/'

file_ptr = open(mapping_file, 'r')
# extract each line into array 'actions'
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    # construct a dictionary: operation_name -> index
    actions_dict[a.split()[1]] = int(a.split()[0])

batch_gen = BatchGenerator(num_classes=len(actions_dict),
                           actions_dict = actions_dict, 
                           gt_path = gt_dir, 
                           features_path = feature_dir, 
                           sample_rate = 1)
batch_gen.read_data(vid_list_file)
batch_input_tensor, batch_target_tensor, mask = batch_gen.next_batch(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_stages = 4      # number of SingleStageModel in MultiStageModel.stages
num_layers = 10     # number of DilatedResidualLayer in SingleStageModel.layers
num_f_maps = 64     # number of kernels in DilatedResidualLayer and conv1d layer
features_dim = 2048 # each feature vector contains 2048 floats
bz = 1              # batch size
lr = 0.0005         # learning rate
num_epochs = 50     # training epoches


model = MultiStageModel(num_stages, num_layers, num_f_maps, features_dim, num_classes=6)
model.to(device)
batch_input_tensor = batch_input_tensor.to(device)
mask = mask.to(device)
predictions = model(batch_input_tensor, mask)
print(predictions)
print(len(batch_input_tensor[0,0]))
print(len(predictions[0,0,0]))






'''
sample_rate = 2
arr = np.array([[1, 2, 3],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15, 16, 17]])
batch_target_tensor = torch.ones(5, 4, dtype=torch.long)*(-100) # ?
print(batch_target_tensor)


# features = np.load("C:/github/casual_tcn/cholec80/train_dataset/video_feature@2020/video14.npy")
# print(np.shape(features)[0])    # number of frames
# print(np.shape(features)[1])    # length of each feature vector

file_ptr = open("./data.txt", 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
("actions")
print(len(actions))
actions_dict = dict()
for action in actions:
    actions_dict[action.split()[1]]=int(action.split()[0])
print("actions_dict")
print(actions_dict)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(2, 2, kernel_size=2)
        self.conv2 = nn.Conv1d(2, 2, kernel_size=3, dilation=1, padding=1)
        # self.conv3 = nn.Conv1d(10, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        print("after conv1: ")
        print(out)

        out = self.conv2(out)
        print("after conv2: ")
        print(out)

        # out = self.conv3(out)
        # print("after conv3")
        # print(out)

        return out

# input = torch.rand(1,2,7)   # 1 videos
#                             # spatial: 5 features in each frame
#                             # temporal: 7 frames
# net = Net()
# print("input: ")
# print(input)
# net(input)

input = torch.rand(2,3,5)
input = (input-0.5)*10
print(input)
print("softmax")
print(F.softmax(input, dim=1))
print("sigmoid")
print(torch.sigmoid(input))

'''
