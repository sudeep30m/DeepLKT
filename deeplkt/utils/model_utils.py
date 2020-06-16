import os
import csv
import torch
import json
from deeplkt.utils.util import make_dir
import numpy as np
from deeplkt.config import *
from torch.utils.data.sampler import SubsetRandomSampler
from shapely.geometry import Polygon
import math

def best_checkpoint(pth, vid=-1):
    curr_max = -1
    for file in os.listdir(pth):
        # print(file[0:4])
        if(file[0] != '.'):
            files = file.split('-')
            if (vid == -1):
                if(files[0] == "best"):

                    curr_max = max(curr_max, int(files[1]))
                    # print(curr_max)
            elif(files[0] == ('v' + str(vid)) and files[1] == "best"):
                curr_max = max(curr_max, int(files[2]))
    return curr_max


def last_checkpoint(pth):
    curr_max = -1
    for file in os.listdir(pth):
        # print(file[0:4])
        if(file[0] != '.' and file[0:4] != 'best'):
            files = file.split('-')
            curr_max = max(curr_max, int(files[0]))
    return curr_max

def img_to_numpy(img):

    img = img.permute(1,2,0)
    return img.detach().cpu().numpy()

def tensor_to_numpy(t):
    # img = img.tranpose(1,2,0)
    return t.detach().cpu().numpy()


def calc_iou(q1, q2):
    p1 = Polygon([(q1[0], q1[1]), (q1[2], q1[3]), (q1[4], q1[5]), (q1[6], q1[7])])
    p2 = Polygon([(q2[0], q2[1]), (q2[2], q2[3]), (q2[4], q2[5]), (q2[6], q2[7])])
    iou = p1.intersection(p2).area / p1.union(p2).area
    return iou


def update_model(model, epoch, optimizer_dict):
    checkpoint_file = model.path + 'checkpoints/' + str(epoch) + ".pt"
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer_dict
    }
    torch.save(state, checkpoint_file)

def load_model(root_folder, folder_name, epoch):
    path = root_folder + folder_name + "/checkpoints/" + str(epoch) +".pt"
    return torch.load(path)


def np_data_to_batches(data, batch_size):
    num_batches = math.ceil(len(data) / batch_size)

    return np.array_split(data, num_batches, 0)


def splitData(dataset, params):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    train_examples = min(dataset_size, params.train_examples)
    split = int(np.floor( (1.0- params.val_split) * train_examples))
    if params.shuffle_train:
        np.random.seed(params.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:train_examples] 
    train_loader = np_data_to_batches(np.array(train_indices), params.batch_size)
    valid_loader = np_data_to_batches(np.array(val_indices), params.batch_size)
    return train_loader, valid_loader

def get_batch(dataset, indices):
    x, y = dataset[0]
    m = len(x)
    data_x = [[] for x in np.arange(m)]
    data_y = []
    for ind in indices:
        x, y = dataset.get_train_data_point(ind)
        for j in range(m):
            data_x[j].append(x[j])
        data_y.append(y)
    data_x[2] = np.array(data_x[2])
    data_x[3] = np.array(data_x[3])
    
    return data_x, np.array(data_y)
