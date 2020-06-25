import os
import glob
from deeplkt.utils.util import make_dir
from deeplkt.utils.bbox import get_min_max_bbox, cxy_wh_2_rect, get_region_from_corner
from deeplkt.configParams import *
import numpy as np
import cv2
import shutil
from os.path import join, isfile
import matplotlib.pyplot as plt
import pandas as pd
from deeplkt.datasets.dataset import *
from deeplkt.utils.model_utils import calc_iou






def convertVideoToDir(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    while True:
        # if(cnt == 0):
        #     ret, frame = cap.read()
        # else:
        ret, frame = cap.read()

        if ret:
            pth = join(output_dir, str(format(cnt, '08d'))+'.jpg')
            # print(pth)
            cv2.imwrite(pth, frame)
            cnt += 1
        else:
            break

def readDir(out_dir):
    imgs = []
    for f in sorted(os.listdir(out_dir)):
        fi = join(out_dir, f)
        img = cv2.imread(fi)
        imgs.append(img)
    return imgs


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame



def convert_frames_to_video(pathIn, pathOut, fps):
    """ Converts all images in pathin directory
        to a video in pathOut path.

    Keyword arguments:
        pathIn -- input images folder
        pathOut -- output video path
        fps -- frames per second of output video
    """

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    # print(files[0][0:-4])
    files.sort(key = lambda x: int(x[0:-4]))
 
    for i in range(len(files)):
        filename= join(pathIn, files[i])
        # print(filename)
        #reading each files
        img = cv2.imread(filename)

        height, width, layers = img.shape
        size = (width,height)
        # print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def draw_bbox(img, bbox, color=(255, 0, 0), thk=1):
    """ Gives an image with bounding box around it.

    Keyword arguments:
        img -- input image
        bbox -- bounding box coordinates 
                x1, y1, x2, y2, x3, y3, x4, y4                 
    Returns:
        Output image with bounding box around it
    """

    bbox = [int(i) for i in bbox]
    for i in range(4):
        cv2.line(img, (bbox[2*i], bbox[2*i + 1]),  \
        (bbox[(2*i + 2) % 8], bbox[(2*i + 3) % 8]), color, thk)
    return img


def outputBboxes(input_images_path, output_images_path, output_file_path):
    """ Forms a folder containing images with
    predicted bounding boxes.

    Keyword arguments:
        input_images_path -- path to folder containing all input images
        output_images_path -- path to folder containing all images with 
                              bounding boxes
        output_file_path -- path to file containing predicte bounding box
                            coordinates
    Returns:
        Generates all images with bboxes in output_images_path folder
    """
    files = glob.glob(output_images_path + '*')
    for f in files:
        os.remove(f)

    img_index = 1
    with open(output_file_path, "r") as filestream:
        
        for line in filestream:
            p_ = line.split(",")
            xbl = (float(p_[0]))
            ybl = (float(p_[1]))
            x1 = (float(p_[2]))
            y1 = (float(p_[3]))
            xtr = (float(p_[4]))
            ytr = (float(p_[5]))
            x2 = (float(p_[6]))
            y2 = (float(p_[7]))
            #     I = [[x1, y1], [x1, y2], [x2, y2], [x2, y1] ] #If rect BB
            I = [x1, y1, xbl, ybl, x2, y2, xtr, ytr] #If free BB
            
            T_PATH = input_images_path +str(format(img_index, '08d'))+'.jpg'
            # print(T_PATH)
            img_t = cv2.imread(T_PATH)
            i_gt = draw_bbox(img_t.copy(), I)
            W_PATH = output_images_path +str(format(img_index, '08d'))+'.jpg'
            cv2.imwrite(W_PATH, i_gt)
            img_index += 1

def visualise_sobel_kernel(kernel):
    img = np.zeros((200, 200, 3)) + 255.0
    font                   = cv2.FONT_HERSHEY_DUPLEX
    fontScale              = 0.5
    fontColor              = (0,0,0)
    lineType               = 1

    for j in range(3):
        for i in range(3):
            cv2.putText(img, str("{0:.2f}".format(kernel[j][i])), 
                (20 + 60 * j, 40 + 60 * i), 
                font, 
                fontScale,
                fontColor,
                lineType)
    # img = cv2.resize(img, (100, 100))

    return img


def visualise_resized_transitions(img_dir, m_dir, out_dir):

    make_dir(out_dir)
    sz = EXEMPLAR_SIZE
    sx = INSTANCE_SIZE
    xmin = sx / 2.0 - sz / 2.0
    xmax = sx / 2.0 + sz / 2.0
    
    i = 0
    while(True):
        if not (os.path.exists(join(m_dir, str(i) +"_tcr.jpeg"))):
            break
        img_tcr = cv2.imread(join(m_dir, str(i) +"_tcr.jpeg"))
        img_i = cv2.imread(join(m_dir, str(i) +"_i.jpeg"))

        dir_path = join(m_dir, str(i) + "-resized")
        out_path = join(out_dir, str(i))
        make_dir(out_path)
        for j, f in enumerate(sorted(os.listdir(dir_path))):
            img = np.zeros((sx, 2*sx, 3))
            quad = np.load(join(dir_path, f))
            img[ 0:sx, sx:2 * sx, :] = draw_bbox(np.array(img_i), quad, color=(0, 255, 0), thk=1)
            # img[ 0:sx, sx:2 * sx, :] = img_i
            img[int(xmin) :  int(xmax), int(xmin) :  int(xmax), :] = img_tcr
            cv2.imwrite(join(out_path, str(j) +".jpeg"), img)
        i += 1

def visualise_transitions(dataset, idx, m1):

    out_video = dataset.get_out_video_path(idx)
    img_dir = join(out_video, "img_tcr")        
    m_dir = join(out_video, m1)
    out_dir = join(out_video, "transitions", m1)        
    make_dir(out_dir)
    sz = EXEMPLAR_SIZE
    sx = INSTANCE_SIZE
    xmin = sx / 2.0 - sz / 2.0
    xmax = sx / 2.0 + sz / 2.0
   
    i = 0
    while(True):
        if not (os.path.exists(join(m_dir, str(i) +"_tcr.jpeg"))):
            break
        img_i_copy = cv2.imread(join(img_dir, str(i) +"_i.jpeg"))
        quad_gt = np.load(join(img_dir, str(i) + "-quad-gt.npy"))
    
        dir_path = join(m_dir, str(i))
        out_path = join(out_dir, str(i))
        make_dir(out_path)
        for j, f in enumerate(sorted(os.listdir(dir_path))):
            quad = np.load(join(dir_path, f))
            img_i = draw_bbox(np.array(img_i_copy), quad, color=(0, 255, 0), thk=1)
            img_i = draw_bbox(img_i, quad_gt, color=(0, 0, 255), thk=1)
            cv2.imwrite(join(out_path, str(j) +".jpeg"), img_i)
        i += 1


def visualise_resized_images(dataset, idx, m1, m2):

    out_video = dataset.get_out_video_path(idx)
    img_dir = join(out_video, "img_tcr")        
    m1_dir = join(out_video, m1)
    m2_dir = join(out_video, m2)
    out2_dir = join(out_video, "resize-results")
    font                   = cv2.FONT_HERSHEY_DUPLEX
    fontScale              = 0.5
    fontColor              = (255,255,255)
    lineType               = 2 

    # make_dir(output_dir)
    make_dir(out2_dir)

    # images = int(len(os.listdir(img_dir)) / 3)

    sz = EXEMPLAR_SIZE
    sx = INSTANCE_SIZE
    xmin = sx / 2.0 - sz / 2.0
    xmax = sx / 2.0 + sz / 2.0
    quad_iden = np.array([xmin, xmax, xmin, xmin, xmax, xmin, xmax, xmax])
    
    i = 0
    while(True):
        img = np.zeros((2*sx, 2*sx, 3))
        if not (os.path.exists(join(m1_dir, str(i) +"_tcr.jpeg"))):
            break
        img_tcr_pure = cv2.imread(join(m1_dir, str(i) +"_tcr.jpeg"))
        img_i_pure = cv2.imread(join(m1_dir, str(i) +"_i.jpeg"))
        img_tcr_learned = cv2.imread(join(m2_dir, str(i) +"_tcr.jpeg"))
        img_i_learned = cv2.imread(join(m2_dir, str(i) +"_i.jpeg"))
        
        quad_pure = np.load(join(m1_dir, str(i) +"-quad-resized.npy"))
        quad_learned = np.load(join(m2_dir, str(i) +"-quad-resized.npy"))

        img_i_pure = draw_bbox(img_i_pure, quad_pure, color=(0, 255, 0), thk=2)
        img_i_pure = draw_bbox(img_i_pure, quad_iden, color=(0, 165, 255), thk=2)
        # draw_bbox(img_i_pure, quad_gt, color=(0, 0, 255), thk=2)

        img_i_learned = draw_bbox(img_i_learned, quad_learned, color=(0, 255, 0), thk=2)
        img_i_learned = draw_bbox(img_i_learned, quad_iden, color=(0, 165, 255), thk=2)
        # draw_bbox(img_i_learned, quad_gt, color=(0, 0, 255), thk=2)
        img[ 0:sx, sx:2 * sx, :] = img_i_pure
        img[int(xmin) :  int(xmax), int(xmin) :  int(xmax), :] = img_tcr_pure
        img[sx:2*sx, sx:2 * sx, :] = img_i_learned
        img[sx + int(xmin) : sx +  int(xmax), int(xmin) :  int(xmax), :] = img_tcr_learned
        cv2.putText(img, "Pure LKT", 
            (90, 220), 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(img, "Learned LKT", 
                    (80, 475), 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)

        # img_i = np.concatenate([img_i_pure, img_i_learned], axis=1)
        cv2.imwrite(join(out2_dir, str(i) +".jpeg"), img)
        i += 1

def visualise_images(dataset, idx, m1, m2):

    out_video = dataset.get_out_video_path(idx)
    img_dir = join(out_video, "img_tcr")        
    m1_dir = join(out_video, m1)
    m2_dir = join(out_video, m2)
    out2_dir = join(out_video, "results")
    
    make_dir(out2_dir)

    images = int(len(os.listdir(img_dir)) / 3)


    i = 0
    while(True):
        if not (os.path.exists(join(img_dir, str(i) +"_i.jpeg"))):
            break
        img_i_pure = cv2.imread(join(img_dir, str(i) +"_i.jpeg"))
        img_i_learned = cv2.imread(join(img_dir, str(i) +"_i.jpeg"))
        
        quad_pure = np.load(join(m1_dir, str(i) +"-quad.npy"))
        # quad_pure_id = np.load(join(m1_dir, str(i) +"-quad-id.npy"))
        quad_learned = np.load(join(m2_dir, str(i) +"-quad.npy"))
        # quad_learned_id = np.load(join(m2_dir, str(i) +"-quad-id.npy"))
        quad_gt = np.load(join(img_dir, str(i) +"-quad-gt.npy"))
        # sz = EXEMPLAR_SIZE
        # sx = INSTANCE_SIZE
        # xmin = sx / 2.0 - sz / 2.0
        # xmax = sx / 2.0 + sz / 2.0
        # quad_iden = np.array([xmin, xmax, xmin, xmin, xmax, xmin, xmax, xmax])

        img_i_pure = draw_bbox(img_i_pure, quad_pure, color=(255, 0, 0), thk=2)
        # draw_bbox(img_i_pure, quad_pure_id, color=(0, 165, 255), thk=2)
        img_i_pure = draw_bbox(img_i_pure, quad_gt, color=(0, 0, 255), thk=2)

        img_i_learned = draw_bbox(img_i_learned, quad_learned, color=(255, 0, 0), thk=2)
        # draw_bbox(img_i_learned, quad_learned_id, color=(0, 165, 255), thk=2)
        img_i_learned = draw_bbox(img_i_learned, quad_gt, color=(0, 0, 255), thk=2)

        img_i = np.concatenate([img_i_pure, img_i_learned], axis=1)
        cv2.imwrite(join(out2_dir, str(i) +".jpeg"), img_i)
        i += 1


def visualise_sobels(dataset, idx, m1, m2):

    out_video = dataset.get_out_video_path(idx)
    img_dir = join(out_video, "img_tcr")        
    m1_dir = join(out_video, m1)
    m2_dir = join(out_video, m2)
    output_dir = join(out_video, "sobels")
    make_dir(output_dir)

    out = []
    images = int(len(os.listdir(img_dir)) / 3)

    font                   = cv2.FONT_HERSHEY_DUPLEX
    fontScale              = 0.5
    fontColor              = (0,0,0)
    lineType               = 1

    i = 0
    mx = 0.0
    idd = 0
    best = 98
    while(True):
        print(i)
        pth = join(m1_dir, str(i) +"_pip_tcr.jpeg")
        if not os.path.exists(pth):
            break
        poster = np.zeros((860, 700, 3)) + 255
        img_tcr = cv2.imread(join(m1_dir, str(i) +"_pip_tcr.jpeg"))
        if(i == best):
            cv2.imwrite("best/best-img_tcr.jpeg", img_tcr)
        img_tcr = cv2.resize(img_tcr, (100, 100))
        img_i = cv2.imread(join(m1_dir, str(i) +"_pip_i.jpeg"))
        quad_pure = np.load(join(m1_dir, str(i) +"_quad_pip.npy"))
        quad_learned = np.load(join(m2_dir, str(i) +"_quad_pip.npy"))
        quad_id = np.load(join(m1_dir, str(i) +"_quad_pip_id.npy"))
        quad_gt = np.load(join(m1_dir, str(i) +"_quad_pip_gt.npy"))
        iou = calc_iou(quad_gt, quad_learned) - calc_iou(quad_gt, quad_pure)
        if(iou >= mx):
            print(mx, iou)
            mx = iou
            idd = i
        img_i = draw_bbox(img_i, quad_pure, color=(0, 255, 255), thk=1)
        img_i = draw_bbox(img_i, quad_learned, color=(0, 255, 0), thk=1)
        img_i = draw_bbox(img_i, quad_gt, color=(0, 0, 255), thk=1)
        img_i = draw_bbox(img_i, quad_id, color=(255, 0, 0), thk=1)
        if(i == best):
            cv2.imwrite("best-imp.jpeg", img_i)
        img_i = cv2.resize(img_i, (200, 200))
                

        poster[100:200, 90:190, :] = img_tcr
        poster[400:600, 40:240, :] = img_i
        
        cv2.putText(poster, "Img tcr", 
            (110, 220), 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.putText(poster, "Img i", 
            (105, 620), 
            font, 
            fontScale,
            fontColor,
            lineType)

        # cv2.putText(poster, "Learned LKT results", 
        #     (65, 820), 
        #     font, 
        #     fontScale,
        #     fontColor,
        #     lineType)
        
        cv2.putText(poster, "Pure LKT", 
            (300, 15), 
            font, 
            fontScale,
            fontColor,
            lineType)
        cv2.putText(poster, "Learned LKT", 
            (410, 15), 
            font, 
            fontScale,
            fontColor,
            lineType)

        cv2.putText(poster, "Learned sobels", 
            (560, 15), 
            font, 
            fontScale,
            fontColor,
            lineType)


        for j in range(3):
            sx = cv2.imread(join(m1_dir, str(i)+"-sx-"+str(j) +".jpeg"))
            if(i == best):
                cv2.imwrite("best-sx-pure-"+str(j)+".jpeg", sx)
            sx = cv2.resize(sx, (100, 100))
            poster[140*j+ 30:140*j+130, 280:380, :] = sx
        for j in range(3, 6):
            sy = cv2.imread(join(m1_dir, str(i)+"-sy-"+str(j-3) +".jpeg"))
            if(i == best):
                cv2.imwrite("best-sy-pure-"+str(j-3)+".jpeg", sy)
            sy = cv2.resize(sy, (100, 100))
            poster[ 140*j+ 30:140*j+130, 280:380,:] = sy
        sx_ker = np.load(join(m2_dir, str(i)+"-sx.npy"))
        sy_ker = np.load(join(m2_dir, str(i)+"-sy.npy"))
        # print(sx_ker.shape, sy_ker.shape)
        for j in range(3):
            sx = cv2.imread(join(m2_dir, str(i)+"-sx-"+str(j) +".jpeg"))
            if(i == best):
                cv2.imwrite("best-sx-learned-"+str(j)+".jpeg", sx)
            sx = cv2.resize(sx, (100, 100))
            poster[ 140*j+ 30:140*j+130, 420:520,:] = sx
            sx_img = visualise_sobel_kernel(sx_ker[j, 0, :, :])
            if(i == best):
                cv2.imwrite("best-sx-learned-sobel-"+str(j)+".jpeg", sx_img)
            sx_img = cv2.resize(sx_img, (100, 100))

            poster[ 140*j+ 30:140*j+130, 560:660,:] = sx_img

        for j in range(3, 6):
            sy = cv2.imread(join(m2_dir, str(i)+"-sy-"+str(j-3) +".jpeg"))
            if(i == best):
                cv2.imwrite("best-sy-learned-"+str(j-3)+".jpeg", sy)
            sy = cv2.resize(sy, (100, 100))
            poster[ 140*j+ 30:140*j+130, 420:520, :] = sy
            sy_img = visualise_sobel_kernel(sy_ker[j-3, 0, :, :])
            if(i == best):
                cv2.imwrite("best-sy-learned-sobel-"+str(j)+".jpeg", sy_img)
            sy_img = cv2.resize(sy_img, (100, 100))
            poster[ 140*j+ 30:140*j+130, 560:660,:] = sy_img
        out.append(poster)
        i += 1
    print("Max idd = ", idd)
    print("Max IOU diff = ", mx)
    writeImagesToFolder(out, output_dir)

def writeImagesToFolder(imgs, folder_dir):
    make_dir(folder_dir)
    for i, img in enumerate(imgs):
        pth = join(folder_dir, str(format(i + 1, '08d'))+'.jpg')
        cv2.imwrite(pth, img)

def visualise_data_point(x, y):
    img_t = x[0]
    img_i = x[1]
    quad_t = x[2]
    quad_i = y
    imgt_box = draw_bbox(img_t, quad_t)
    imgi_box = draw_bbox(img_i, quad_i)
    vis = np.concatenate((imgt_box, imgi_box), axis=1)
    return vis

def get_subwindow(im, pos, model_sz, original_sz, avg_chans, ind=-1):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """
    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    # context_xmin = round(pos[0] - c) # py2 and py3 round
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    # context_ymin = round(pos[1] - c)
    context_ymin = np.floor(pos[1] - c + 0.5)
    context_ymax = context_ymin + sz - 1
    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    r, c, k = im.shape
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
        te_im = np.zeros(size, np.uint8)
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                            int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch = im[int(context_ymin):int(context_ymax + 1),
                        int(context_xmin):int(context_xmax + 1), :]
    # return im_patch
    if not np.array_equal(model_sz, original_sz):
        im_patch = cv2.resize(im_patch, (model_sz, model_sz))
    return im_patch


def visualise_transformed_data_point(x, y):
    img_t = x[0]
    img_i = x[1]
    bbox_t = x[2][np.newaxis, :]
    bbox_t = get_min_max_bbox(bbox_t)
    bbox_t = cxy_wh_2_rect(bbox_t)

    center_pos = np.array([bbox_t[0, 0]+(bbox_t[0, 2]-1)/2.0,
                                bbox_t[0, 1]+(bbox_t[0, 3]-1)/2.0])
    size = np.array([bbox_t[0, 2], bbox_t[0, 3]])

    # calculate z crop size
    w_z = size[0] + CONTEXT_AMOUNT * np.sum(size)
    h_z = size[1] + CONTEXT_AMOUNT * np.sum(size)
    s_z = np.round(np.sqrt(w_z * h_z))
    scale_z = EXEMPLAR_SIZE / s_z
    s_x = np.round(s_z * (INSTANCE_SIZE / EXEMPLAR_SIZE))

        # print("Track centre = ", center_pos)
    channel_average = np.mean(img_t, axis=(0, 1))
    # get crop
    # cv2.imwrite('/home/sudeep/Desktop/img1.jpg', img)
    # print(img.shape)
    # print("Init centre = ", center_pos)
    img_t = get_subwindow(img_t, center_pos,
                                INSTANCE_SIZE,
                                s_x, channel_average)
    sz = EXEMPLAR_SIZE
    sx = INSTANCE_SIZE
    centre = np.array([(sx / 2.0), (sx / 2.0)])
    xmin = centre[0] - (sz / 2.0)
    xmax = centre[0] + (sz / 2.0)
    t_quad = np.array([xmin, xmax, xmin, xmin, xmax, xmin, xmax, xmax]) #inclusive
    img_i = get_subwindow(img_i, center_pos,
                                INSTANCE_SIZE,
                                s_x, channel_average)
    i_quad = y
    imgt_box = draw_bbox(img_t, t_quad)
    imgi_box = draw_bbox(img_i, i_quad)
    vis = np.concatenate((imgt_box, imgi_box), axis=1)
    return vis

    # b = torch.Tensor(BATCH_SIZE, EXEMPLAR_SIZE, EXEMPLAR_SIZE)
    # z_crop = torch.cat(z_crop)        # print(z_crop)
    # temp = img_to_numpy(z_crop[0])
    # print(temp.shape)
    # cv2.imwrite("temp.jpeg", temp)

def _min(arr):
    return np.min(np.array(arr), 0)

def _max(arr):
    return np.max(np.array(arr), 0)


def _bbox_clip(cx, cy, width, height, boundary):

    cx = _max([0, _min([cx, boundary[1]]) ])
    cy = _max([0, _min([cy, boundary[0]]) ])
    width = _max([10, _min([width, boundary[1]]) ])
    height = _max([10, _min([height, boundary[0]]) ])
    return cx, cy, width, height


def transform_to_gt(x, y):
    img_t = x[0]
    img_i = x[1]
    bbox_t = x[2][np.newaxis, :]
    bbox_t = get_min_max_bbox(bbox_t)
    bbox_t = cxy_wh_2_rect(bbox_t)
    y = get_min_max_bbox(y[np.newaxis, :])[0]

    center_pos = np.array([bbox_t[0, 0]+(bbox_t[0, 2]-1)/2.0,
                                bbox_t[0, 1]+(bbox_t[0, 3]-1)/2.0])
    size = np.array([bbox_t[0, 2], bbox_t[0, 3]])

    # calculate z crop size
    w_z = size[0] + CONTEXT_AMOUNT * np.sum(size)
    h_z = size[1] + CONTEXT_AMOUNT * np.sum(size)
    s_z = np.round(np.sqrt(w_z * h_z))
    scale_z = EXEMPLAR_SIZE / s_z
    y[0] -= (INSTANCE_SIZE / 2)
    y[1] -= (INSTANCE_SIZE / 2)
    y[2] -= (EXEMPLAR_SIZE)
    y[3] -= (EXEMPLAR_SIZE)
    
    y = y / scale_z
    # print("Bounding box shape = ", bbox.shape)
    

    # lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

    cx = y[0] + center_pos[0]
    cy = y[1] + center_pos[1]
    # bbox2 = [x.detach() for x in bbox]
    # # smooth bbox
    # print(size, bbox)
    width = size[0] * (1 - TRANSITION_LR) + (size[0] + y[2]) * TRANSITION_LR
    height = size[1] * (1 - TRANSITION_LR) + (size[1]+ y[3]) * TRANSITION_LR
    cx, cy, width, height = _bbox_clip(cx, cy, width,
                                                height, img_i.shape[:2])

    bbox = np.array([cx - width / 2,
            cy - height / 2,
            width,
            height])
    quad_num = get_region_from_corner(bbox[np.newaxis, :])[0]
    return quad_num

def plot_bar_graph(results, path):
    cmap = plt.get_cmap('tab10')
    for k in results:
        num = len(results[k])
    # print(num)
    results['x'] = np.array(range(1, num + 1))
    colors = [cmap(i) for i in np.linspace(0, 1, len(results))]

    df=pd.DataFrame(results)
    df['sort_val'] = df.pure_lkt - df.learned_lkt
    df = df.sort_values('sort_val').drop('sort_val', 1)
    pos = np.arange(num)
    bar_width = 0.3
    leg = []
    for (i, label) in enumerate(results):
        if(label == 'x'):
            continue
        leg.append(label)

    for (i, label) in enumerate(results):
        if(label == 'x'):
            continue
        plt.bar(pos + i*bar_width, label, bar_width, data=df, color=cmap(i), edgecolor='black')
        # plt.plot( 'x', label, data=df, c = cmap(i))

    plt.legend(leg)
    plt.xticks(pos + 0.1, df['x'])
    plt.ylim(0.6, 1.0)
    plt.xlabel('VOT sequence')
    plt.ylabel('IOU')
    plt.title('Pairwise IOU Pure LKT vs Learned Sobel LKT')
    plt.savefig(path)

def plot_different_results(results, path):
    cmap = plt.get_cmap('tab10')
    for k in results:
        # print(k)
        if(k != 'x'):
            num = len(results[k])
    # print(num)
    results['x'] = np.array(range(1, num + 1))
    colors = [cmap(i) for i in np.linspace(0, 1, len(results))]

    df=pd.DataFrame(results)

    for (i, label) in enumerate(results):
        # print(label)
        if(label == 'x'):
            continue
        plt.plot( 'x', label, data=df, c = cmap(i))

    plt.legend()
    plt.xlabel('VOT sequence')
    plt.ylabel('IOU')
    plt.title('Pairwise IOU Pure LKT vs VGG learned sobel LKT')

    plt.savefig(path)


if __name__ == '__main__':
    # pth = '/home/sudeep/Documents/mtp/lkt/data/VOT/VOT_images/VOT_02'
    # convert_frames_to_video(pth, '/home/sudeep/Documents/mtp/pysot/demo/basketball.avi', 30)


    vot_root_dir = '../../data/VOT/'
    alov_root_dir = '../../data/ALOV/'

    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    vot = VotDataset(os.path.join(vot_root_dir,
                        'VOT_images/'),
                    os.path.join(vot_root_dir,
                        'VOT_ann/'),
                    os.path.join(vot_root_dir,
                        'VOT_results/')
                    )

    alov = AlovDataset(os.path.join(alov_root_dir,
                        'ALOV_images/'),
                    os.path.join(alov_root_dir,
                        'ALOV_ann/'),
                    os.path.join(alov_root_dir,
                        'ALOV_results/')
                        )

    results = '../vot/'
    for i in range(1000):
        x, y = vot.get_train_data_point(i)
        img = visualise_transformed_data_point(x, y)
        i_quad = transform_to_gt(x, y)
        img_t = x[0]
        img_i = x[1]
        t_quad = x[2]
        imgt_box = draw_bbox(img_t, t_quad)
        imgi_box = draw_bbox(img_i, i_quad)
        vis = np.concatenate((imgt_box, imgi_box), axis=1)

        cv2.imwrite(results + str(i) +".jpeg", img)
        cv2.imwrite(results + "orig-" + str(i) +".jpeg", vis)