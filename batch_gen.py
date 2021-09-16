#!/usr/bin/python2.7

import torch
import numpy as np
import random


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()      # containing all the video names to be used
        self.index = 0                      # basically counting how many videos have been used
        self.num_classes = num_classes      # number of operations
        self.actions_dict = actions_dict    # dict: operation name -> index
        self.gt_path = gt_path              # directory to ground truch
        self.features_path = features_path  # directory to video features
        self.sample_rate = sample_rate      # sample rate = 1 by default

    def reset(self):                        # manually shuffle the order of video names
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):                     # ?
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):     # read the video names and shuffle the order
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size] # 'batch' is a lsit of video names in this batch
        self.index += batch_size    # update how many videos have been used

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy') # load the feature vectors of this video
            features = np.transpose(features)                                   # 1st order is channel  (spatial)
                                                                                # 2nd order is frames   (terporal)
            file_ptr = open(self.gt_path + vid.split('.')[0] + '.txt', 'r')                            
            content = file_ptr.read().split('\n')[:-1]                          # content: ground truth list (in order)
            for i in range(len(content)):
                content[i] = content[i].split("\t")[1]
            file_ptr.close()
            classes = np.zeros(min(np.shape(features)[1], len(content)))        # construct gt list with proper length
                                                                                # choose the smaller one from:
                                                                                #   number of feature vectors
                                                                                #   number of ground truth values
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]                      # each operation is represented by index
            batch_input .append(features[:, ::self.sample_rate])                # 2D syntax: classes[1st order, 2nd order]
            batch_target.append(classes[::self.sample_rate])                    # 1D syntax: classes[start:end:interval]

        ## create the tensors for torch computation ##
        max_length = max(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max_length, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max_length, dtype=torch.long)*(-100) # why -100 ?
        mask = torch.zeros(len(batch_input), self.num_classes, max_length, dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
