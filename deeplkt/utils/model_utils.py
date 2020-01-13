import os
import csv
import torch
import json
from deeplkt.utils.util import make_dir
import numpy as np
from deeplkt.config import *
from torch.utils.data.sampler import SubsetRandomSampler
from shapely.geometry import Polygon


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


def splitData(dataset):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    train_examples = min(dataset_size, TRAIN_EXAMPLES)
    split = int(np.floor(VALIDATION_SPLIT * train_examples))
    if SHUFFLE_TRAIN:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:train_examples], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                                    sampler=valid_sampler)
    return train_loader, validation_loader