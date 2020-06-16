import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from os.path import isfile, join
import pickle as pkl

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

# def stat_cuda(msg):
#     print('--', msg)
#     print('allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
#         torch.cuda.memory_allocated() / 1024 / 1024,
#         torch.cuda.max_memory_allocated() / 1024 / 1024,
#         torch.cuda.memory_cached() / 1024 / 1024,
#         torch.cuda.max_memory_cached() / 1024 / 1024
#     ))

def pkl_save(pth, obj):
    f = open(pth, "wb")
    pkl.dump(obj, f)
    f.close()

def pkl_load(pth):
    f = open(pth, "rb")
    obj = pkl.load(f)
    return obj

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def trf(img):
    if(len(img.shape)) == 2:
#         print("Yeah")
        return img
    return img[..., ::-1]

def draw(images, cols):
    n = len(images)
    r = int(n / cols)
    if (n % cols) != 0:
        r += 1 
    c = cols
    fig, axs = plt.subplots(nrows=r, ncols=c, figsize=(50, 30),
                            subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)
    
    for i in range(len(images)):
        if(len(images[i].shape) == 2):
            (axs.flat)[i].imshow(trf(images[i]), cmap='gray')
        else:
            (axs.flat)[i].imshow(trf(images[i]))
        
def warp(x, y, p):
    return np.array([(1. + p[0]) * x + p[2] * y + p[4], p[1] * x + (1 + p[3]) * y + p[5]], dtype='float32')



def get_warped_corners(x1, y1, x2, y2, p):
    t1 = warp(x1, y1, p)
    t2 = warp(x1, y2, p)
    t3 = warp(x2, y2, p)
    t4 = warp(x2, y1, p)
    return [t1, t2, t3, t4] 


def get_image(folder_path, i):
    path = folder_path  + str(format(i + 1, '08d'))+'.jpg'
    return cv2.imread(path)
        
def get_rectangle(file_path, i):
    with open(file_path, "r") as filestream:
        lines = filestream.readlines()
        p_ = lines[i].split(",")
        x1 = (float(p_[2]))
        y1 = (float(p_[3]))
        x2 = (float(p_[6]))
        y2 = (float(p_[7]))
    return x1, y1, x2, y2

def get_quadrilateral(file_path, i):
    with open(file_path, "r") as filestream:
        lines = filestream.readlines()
        p_ = lines[i].split(",")
        xbl = (float(p_[0]))
        ybl = (float(p_[1]))
        x1 = (float(p_[2]))
        y1 = (float(p_[3]))
        xtr = (float(p_[4]))
        ytr = (float(p_[5]))
        x2 = (float(p_[6]))
        y2 = (float(p_[7]))
    return xbl,ybl,x1, y1,xtr,ytr, x2, y2
        
def enlarge(x1, y1, x2, y2, ratio):
    
    dy = float(y2 - y1) * (ratio - 1.0) * 0.5
    dx = float(x2 - x1) * (ratio - 1.0) * 0.5
    
    return x1 - dx, y1 - dy, x2 + dx, y2 + dy

def crop(img, x1, y1, x2, y2):
        
    return img[int(y1):int(y2), int(x1):int(x2)]

def write_to_output_file(rectangles, file_path):
    outfile_2 = open(file_path, "w")
    for rect in rectangles:
        [xbl,ybl,x1, y1,xtr,ytr, x2, y2] = rect
        outfile_2.write(str(xbl)+','+str(ybl)+','+str(x1)+','+str(y1)+','+str(xtr)+','+str(ytr)+','+str(x2)+','+str(y2)+ '\n')
    outfile_2.close()

def get_ground_truth_p(input_file_path, i):
    x1, y1, x2, y2 = get_rectangle(input_file_path, i)
    x3, y3, x4, y4 = get_rectangle(input_file_path, i + 1)
    pts1 = np.float32([[x1, y1], [x1, y2], [x2, y2]])
    pts2 = np.float32([[x3, y3], [x3, y4], [x4, y4]])
    M = cv2.getAffineTransform(pts1, pts2)
    p_ground_truth = np.float32([M[0][0] - 1.0, M[1][0], M[0][1],  M[1][1] - 1.0, M[0][2], M[1][2]])
    return p_ground_truth


def analyse_output(out):
    if(isinstance(out, list)):
        for ele in out:
            analyse_output(ele)
    else:
        print("Shape of output = ", out.shape)
        print("Norm of output = " , torch.norm(out))
        print("Mean of output = ", torch.mean(out))
        print("Variance of output = ", torch.var(out))
        print()
        print(out)

