import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random

# 40 videos in total
train_vid = [1,40]
test_vid = [41,80]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')

args = parser.parse_args()

num_stages = 4      # number of SingleStageModel in MultiStageModel.stages
num_layers = 10     # number of DilatedResidualLayer in SingleStageModel.layers
num_f_maps = 64     # number of kernels in DilatedResidualLayer and conv1d layer
features_dim = 2048 # each feature vector contains 2048 floats
bz = 1              # batch size
lr = 5.0e-4         # learning rate
num_epochs = 20     # training epoches

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

vid_list_file = 'C:/github/casual_tcn/cholec80/train.txt'           # video list file for training
vid_list_file_tst = 'C:/github/casual_tcn/cholec80/test.txt'        # video list file for testing
features_path = 'C:/github/casual_tcn/cholec80/features/'           # directory to video features
gt_path = 'C:/github/casual_tcn/cholec80/groundtruth/'              # directory to ground truth 
mapping_file = 'C:/github/casual_tcn/cholec80/action_mapping.txt'   # a mapping from indices to operations

model_dir = "./models/"      # the directory to save result models
results_dir = "./results/"   # the directory to save predicted results
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# update train video list
if os.path.exists(vid_list_file):
    os.remove(vid_list_file)
file_ptr = open(vid_list_file, "w")
for i in range(train_vid[0], train_vid[1]+1):
    file_ptr.write("video%02d.mp4\n" % (i))
file_ptr.close()

# update test video list
if os.path.exists(vid_list_file_tst):
    os.remove(vid_list_file_tst)
file_ptr = open(vid_list_file_tst, "w")
for i in range(test_vid[0], test_vid[1]+1):
    file_ptr.write("video%02d.mp4\n" % (i))
file_ptr.close()

# construct action dictionary: oepration_name -> index
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1] # extract each line into array 'actions'
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0]) # construct a dictionary: operation_name -> index

num_classes = len(actions_dict)

trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

the_epoch = 37
if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
