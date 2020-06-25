import os
import csv
import torch
import json
from deeplkt.utils.util import make_dir
import numpy as np
from deeplkt.configParams import *
from torch.utils.data.sampler import SubsetRandomSampler
from shapely.geometry import Polygon
import math

# def save_model(model, optimizer_dict, root_path, folder_name):
#     folder_path = root_path + folder_name + "/"
#     make_dir(folder_path)
#     make_dir(folder_path + "checkpoints/")
#     checkpoint_file = folder_path + 'checkpoints/0' + ".pt"
#     state = {
#         'epoch': 0,
#         'state_dict': model.state_dict(),
#         'optimizer' : optimizer_dict
#     }

#     torch.save(state, checkpoint_file)

#     json_f = open(folder_path + 'params.json', 'w')
#     json.dump(model.params, json_f)
#     json_f.close()
#     model.path = folder_path
#     # lines[2] = ro

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

# def save_results(path, res):
#     f = open(path, "w")
#     for i in range(len(res)):
#         f.write(str(res[i]) +"\n")
#     f.close()

# def load_results(root_folder, folder_name):
#     path = root_folder + folder_name + "/" + "results.txt"
#     f = open(path, "r")
#     lines = f.readlines()
#     lines = [int(r) for r in lines]
#     return np.array(lines)

# def dump_res(results):
#     f = open('results.pb', 'wb')
#     pkl.dump(results, f)
#     f.close()

# def load_res(results):
#     f = open('results.pb', 'rb')
#     results = pkl.load(f)
#     f.close()
#     return results


def np_data_to_batches(data, batch_size):
    # data = []
    # for x in dataset:
    #     data.append(x)
    # data = np.array(data)
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

    # Creating PT data samplers and loaders:
    # train_sampler = SubsetRandomSampler(train_indices)
    # valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = np_data_to_batches(np.array(train_indices), params.batch_size)
    valid_loader = np_data_to_batches(np.array(val_indices), params.batch_size)
    return train_loader, valid_loader

def get_batch(dataset, indices):
    # n = len(indices)
    x, y = dataset[0]
    m = len(x)
    # print(m)
    data_x = [[] for x in np.arange(m)]
    data_y = []
    for ind in indices:
        # print(ind)
        x, y = dataset.get_train_data_point(ind)
        # print(x[0].shape, x[1].shape, x[2].shape, x[3].shape)
        # print(y)
        for j in range(m):
            data_x[j].append(x[j])
        data_y.append(y)
    # print(indices)
    # print([np.array(x).shape for x in data_x])
    # try:
    #     data_x = [np.array(x) for x in data_x]
    # except:
    #     from IPython import embed;embed()
    data_x[2] = np.array(data_x[2])
    data_x[3] = np.array(data_x[3])
    
    return data_x, np.array(data_y)

# def splitData(dataset, batch_size, shuffle):
#     # data = []
#     # for x in dataset:
#     #     data.append(x)
#     data = np.array(data)
#     train_examples = min(len(dataset), TRAIN_EXAMPLES)
#     split = int(np.floor(VALIDATION_SPLIT * train_examples))

#     if(shuffle):
#         np.random.seed(RANDOM_SEED)
#         np.random.shuffle(data)
#     train_data, val_data = data[split:train_examples], data[:split]

#     # Creating PT data samplers and loaders:
#     # train_sampler = SubsetRandomSampler(train_indices)
#     # valid_sampler = SubsetRandomSampler(val_indices)

#     train_loader = np_data_to_batches(train_data, batch_size)
#     valid_loader = np_data_to_batches(valid_sampler, batch_size)
#     return train_loader, valid_loader