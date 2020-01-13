import os
import glob
from deeplkt.utils.util import make_dir
import numpy as np
import cv2
import shutil
from os.path import join, isfile

def convertVideoToDir(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    while True:
        ret, frame = cap.read()
        if ret:
            pth = join(output_dir, str(format(cnt, '08d'))+'.jpg')
            print(pth)
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

    img_index = 2 
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
            
            T_PATH = input_images_path +str(format(img_index, '0'))+'.jpg'
            img_t = cv2.imread(T_PATH)
            i_gt = draw_bbox(img_t.copy(), I)
            W_PATH = output_images_path +str(format(img_index, '08d'))+'.jpg'
            cv2.imwrite(W_PATH, i_gt)
            img_index += 1

if __name__ == '__main__':
    pth = '/home/sudeep/Documents/mtp/lkt/data/VOT/VOT_images/VOT_02'
    convert_frames_to_video(pth, '/home/sudeep/Documents/mtp/pysot/demo/basketball.avi', 30)