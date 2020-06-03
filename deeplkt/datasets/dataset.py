import os
import numpy as np
import cv2
import random
from shapely.geometry import Polygon
import torch
from torch.utils.data import Dataset, DataLoader
from deeplkt.utils.visualise import draw_bbox, visualise_sobels, \
    visualise_images, visualise_resized_images, visualise_transitions,\
    visualise_resized_transitions
from deeplkt.utils.bbox import cxy_wh_2_rect, get_min_max_bbox, get_region_from_center
from deeplkt.config import *
from os.path import join
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LKTDataset(Dataset):

    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.inp_ids = []
        self.out_ids = []
        self.index_dict = {}
        self.video_dict = {}
        self.num_samples = 0


    def get_in_video_path(self, vidx):
        return self.inp_ids[vidx]

    def get_out_video_path(self, vidx):
        return self.out_ids[vidx]

    def get_num_images(self, vidx):
        return len(self.x[vidx])
    
    def get_quad(self, ann):
        return np.ones((8, ))
        
    def __len__(self):
        return self.num_samples
    
    def get_video_id(self, ind):
        return self.index_dict[ind]
    
    def __getitem__(self, index):
        vidx, idx = self.index_dict[index]
        point =  self.get_point(vidx, idx)
        # point = [p.unsqueeze(0) for p in point]
        return point[:-1], point[-1]

    def visualise_transitions(self, idx, m1):

        out_video = self.get_out_video_path(idx)
        imgs_in_dir = join(out_video, "img_tcr")        
        m1_in_dir = join(out_video, m1)
        trans_out_dir = join(out_video, "transitions", m1)        
        visualise_transitions(imgs_in_dir, m1_in_dir, trans_out_dir)

    def visualise_transitions_resized(self, idx, m1):

        out_video = self.get_out_video_path(idx)
        imgs_in_dir = join(out_video, "img_tcr")        
        m1_in_dir = join(out_video, m1)
        trans_out_dir = join(out_video, "transitions-resized", m1)        
        visualise_resized_transitions(imgs_in_dir, m1_in_dir, trans_out_dir)

    def visualise_sobels(self, idx, m1, m2):

        out_video = self.get_out_video_path(idx)
        imgs_out_dir = join(out_video, "img_tcr")        
        m1_out_dir = join(out_video, m1)
        m2_out_dir = join(out_video, m2)
        sobel_out_dir = join(out_video, "sobels")
        
        visualise_sobels(imgs_out_dir, m1_out_dir, m2_out_dir, sobel_out_dir)

    def visualise_images(self, idx, m1, m2):

        out_video = self.get_out_video_path(idx)
        imgs_out_dir = join(out_video, "img_tcr")        
        m1_out_dir = join(out_video, m1)
        m2_out_dir = join(out_video, m2)
        results_out_dir = join(out_video, "results")
        
        visualise_images(imgs_out_dir, m1_out_dir, m2_out_dir, results_out_dir)

    def visualise_resized_images(self, idx, m1, m2):

        out_video = self.get_out_video_path(idx)
        imgs_out_dir = join(out_video, "img_tcr")        
        m1_out_dir = join(out_video, m1)
        m2_out_dir = join(out_video, m2)
        results_out_dir = join(out_video, "resize-results")
        
        visualise_resized_images(imgs_out_dir, m1_out_dir, m2_out_dir, results_out_dir)

    def get_orig_sample(self, vid_idx, idx, i=1):
        """
        Returns original image with bounding box at a specific index.
        Range of valid index: [0, self.len-1].
        """
        curr = cv2.imread(self.x[vid_idx][idx][i])
        currbb = self.get_quad(self.y[vid_idx][idx][i])
        sample = {'image': curr, 'bb': currbb}
        return sample
    


    def show_image_with_gt(self, vidx, idx):
        sample = self.get_orig_sample(vidx, idx, i=0)
        img = sample['image']
        box = sample['bb']
        # print(box)
        img_box = draw_bbox(img, box, color=(0, 0, 255))
        return img_box        

    def save_gtbox_video(self, vidx, pathOut, fps=15):
        frame_array = []
        num_imgs = self.get_num_images(vidx)    
        for i in range(num_imgs):
            img = self.show_image_with_gt(vidx, i)
            height, width, layers = img.shape
            size = (width,height)
            frame_array.append(img)
 
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
        for i in range(len(frame_array)):
            out.write(frame_array[i])
        out.release()



    def get_point(self, vid_idx, idx):
        curr_sample = self.get_orig_sample(vid_idx, idx, 1)
        prev_sample = self.get_orig_sample(vid_idx, idx, 0)
        curr_img  = curr_sample['image']
        prev_img  = prev_sample['image']
        curr_box  = curr_sample['bb']
        prev_box  = prev_sample['bb']
        p_init = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        data = [prev_img, curr_img, prev_box, p_init, curr_box]
        # data[0] = self.transform()(data[0]) * 255.0
        # data[1] = self.transform()(data[1]) * 255.0
        # data[0] = torch.from_numpy(data[0]).float().permute(2, 0, 1)
        # data[1] = torch.from_numpy(data[1]).float().permute(2, 0, 1)

        # data[0] = torch.from_numpy(data[0]).to(self.device).float().permute(2, 0, 1)
        # data[1] = torch.from_numpy(data[1]).to(self.device).float().permute(2, 0, 1)        
        # data[2] = torch.from_numpy(data[2]).to(self.device).float()
        # data[3] = torch.from_numpy(data[3]).to(self.device).float()
        # data[4] = torch.from_numpy(data[4]).to(self.device).float()
        
        return data


    def get_modified_target(self, x, bbox_gt):
        bbox_gt = bbox_gt[np.newaxis, :]
        bbox = x[2][np.newaxis, :]
        y_gt = get_min_max_bbox(bbox_gt)
        y = get_min_max_bbox(bbox)
        # print("BBox = ", bbox)
        # print("Y = ", y)

        size = np.array([y[:, 2], y[:, 3]])
        size = size.transpose()
        w_z = size[:, 0] + CONTEXT_AMOUNT * np.sum(size, 1)
        h_z = size[:, 1] + CONTEXT_AMOUNT * np.sum(size, 1)
        s_z = np.sqrt(w_z * h_z)
        scale_z = NEW_EXEMPLAR_SIZE / s_z
        # print(w_z, h_z, s_z, scale_z)
        y_gt -= y
        y_gt = y_gt * scale_z

        y_gt[:, 0] += (NEW_INSTANCE_SIZE / 2)
        y_gt[:, 1] += (NEW_INSTANCE_SIZE / 2)
        y_gt[:, 2] += (NEW_EXEMPLAR_SIZE)
        y_gt[:, 3] += (NEW_EXEMPLAR_SIZE)
        y_gt = get_region_from_center(y_gt)
        return y_gt[0]        
        
    # def get_train_data_point(self, ind):
    #     vid, idx = self.get_video_id(ind)
    #     return self.get_train_data_point(vid, idx)

    def get_train_data_point(self, *args):
        if(len(args) == 1):
            vid, idx = self.get_video_id(args[0])
        else:
            vid, idx = args
    
        x, y = self.get_data_point(vid, idx)
        # print([i.shape for i in x])
        # print(y.shape)
        # print("Original y = ", y)
    
        y = self.get_modified_target(x, y)
        # print("Modified y = ", y)
        # dp = [ele.unsqueeze(0) for ele in dp]
        # bbox = get_min_max_bbox(dp[-1])
        # bbox = cxy_wh_2_rect(bbox)
        # y = torch.from_numpy(y).to(self.device).float()
        return x, y


    def get_data_point(self, vid_idx, idx):
        dp = self.get_point(vid_idx, idx)
        # dp = [ele.unsqueeze(0) for ele in dp]
        # bbox = get_min_max_bbox(dp[-1])
        # bbox = cxy_wh_2_rect(bbox)

        return dp[:-1], dp[-1]


    def get_data(self, vid_idx, shuffle=False):

        num_samples = len(self.x[vid_idx])
        database = []
        for idx in range(num_samples):
            data  = self.get_point(vid_idx, idx)
            database.append(data)

        if(shuffle):
            random.shuffle(database)

        # print(len(data_set))
        images_t_tr = np.array([row[0] for row in database])
        images_i_tr = np.array([row[1] for row in database])
        rects_tr = np.array([row[2] for row in database])
        ps_tr = np.array([row[3] for row in database])
        rects_gt_tr = np.array([row[4] for row in database])
        database_x = [images_t_tr, images_i_tr, rects_tr, ps_tr]
        database_y = rects_gt_tr
        return database_x, database_y

    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)




class AlovDataset(LKTDataset):

    def __init__(self, root_dir, annot_dir, result_dir, device):
        self.exclude = ['01-Light_video00016',
                        '01-Light_video00022',
                        '01-Light_video00023',
                        '02-SurfaceCover_video00012',
                        '03-Specularity_video00003',
                        '03-Specularity_video00012',
                        '10-LowContrast_video00013']
        self.index_dict = {}
        self.video_dict = {}

        self.x, self.y, self.z, self.inp_ids, self.out_ids = self._parse_data(root_dir, annot_dir, result_dir)
        self.num_videos = len(self.x)
        self.device = device


    def _parse_data(self, root_dir, anno_dir, result_dir):
        """
        Parses ALOV dataset and builds tuples of (template, search region)
        tuples from consecutive annotated frames.
        """
        x_videos = []
        y_videos = []
        z_videos = []
        inp_ids = []
        out_ids = []
        envs = os.listdir(root_dir)
        num_anno = 0
        print('Parsing ALOV dataset...')
        for env in envs:
            env_videos = os.listdir(root_dir + env)
            for vid in env_videos:
                if vid in self.exclude:
                    continue
                x = []
                y = []
                z = []
                vid_src = root_dir + env + "/" + vid
                vid_ann = anno_dir + env + "/" + vid + ".ann"
                vid_res = result_dir + env + "/" + vid
                self.make_dir(vid_res)
                self.make_dir(vid_res + "/images")
                frames = os.listdir(vid_src)
                frames.sort()
                frames_inp = [vid_src + "/" + frame for frame in frames]
                frames_out = [vid_res + "/images/" + frame for frame in frames]

                f = open(vid_ann, "r")
                annotations = f.readlines()
                f.close()
                frame_idxs = [int(ann.split(' ')[0])-1 for ann in annotations]
                frames_inp = np.array(frames_inp)
                num_anno += len(annotations)

                for i in range(len(frame_idxs)-1):
                    idx = frame_idxs[i]
                    next_idx = frame_idxs[i+1]
                    x.append([frames_inp[idx], frames_inp[next_idx]])
                    y.append([annotations[i], annotations[i+1]])
                    z.append([frames_out[idx], frames_out[next_idx]])
                    
                    
                x_videos.append(x)
                y_videos.append(y)
                z_videos.append(z)
                inp_ids.append(vid_src)
                out_ids.append(vid_res)

        x_videos = np.array(x_videos)
        y_videos = np.array(y_videos)
        # self.len = len(y)
        print('ALOV dataset parsing done.')
        print('Total number of videos in ALOV dataset = %d' % (len(x_videos)))
        print('Total number of annotations in ALOV dataset = %d' % (num_anno))
        self.num_samples = 0
        for i in range(len(x_videos)):
            for j in range(len(x_videos[i])):
                self.index_dict[self.num_samples] = (i, j)
                self.video_dict[(i, j)] = self.num_samples
                self.num_samples += 1
        return x_videos, y_videos, z_videos, inp_ids, out_ids


    def get_quad(self, ann):
        """
        Parses ALOV annotation and returns bounding box in the format:
        [left, upper, width, height]
        """
        p_ = ann.strip().split(' ')
        xbl = (float(p_[1]))
        ybl = (float(p_[2]))
        x1 = (float(p_[3]))
        y1 = (float(p_[4]))
        xtr = (float(p_[5]))
        ytr = (float(p_[6]))
        x2 = (float(p_[7]))
        y2 = (float(p_[8]))
        return np.array([xbl, ybl, x1, y1, xtr, ytr, x2, y2])

    def get_num_videos(self):
        return self.num_videos


# alov = ALOVDataset(os.path.join(args.data_directory,
#                     'imagedata++/'),
#                     os.path.join(args.data_directory,
#                     'alov300++_rectangleAnnotation_full/')
#                     )


class VotDataset(LKTDataset):

    def __init__(self, root_dir, annot_dir, result_dir, device):
        self.index_dict = {}
        self.video_dict = {}


        self.x, self.y, self.z, self.inp_ids, self.out_ids = self._parse_data(root_dir, annot_dir, result_dir)
        self.num_videos = len(self.x)
        self.device = device


    def _parse_data(self, root_dir, anno_dir, result_dir):
        """
        Parses VOT dataset and builds tuples of (template, search region)
        tuples from consecutive annotated frames.
        """
        x_videos = []
        y_videos = []
        z_videos = []
        out_ids = []
        inp_ids = []
        num_anno = 0
        print('Parsing VOT dataset...')
        videos = os.listdir(root_dir)
        videos.sort()
        for vid in videos:
            # print(vid)
            x = []
            y = []
            z = []
            vid_src = root_dir + vid
            vid_ann = anno_dir + vid + "/groundtruth.txt"
            vid_res = result_dir + vid
            self.make_dir(vid_res)
            self.make_dir(vid_res + "/images")

            frames = os.listdir(vid_src)
            frames.sort()
            frames_inp = [vid_src + "/" + frame for frame in frames]
            frames_out = [vid_res + "/images/" + frame for frame in frames]
            
            f = open(vid_ann, "r")
            annotations = f.readlines()
            f.close()
            for i in range(len(frames) - 1):
                x.append([frames_inp[i], frames_inp[i + 1]])
                y.append([annotations[i], annotations[i + 1]])
                z.append([frames_out[i], frames_out[i + 1]])
                num_anno += 1
            x_videos.append(x)
            y_videos.append(y)
            z_videos.append(z)
            inp_ids.append(vid_src)
            out_ids.append(vid_res)
            
        x_videos = np.array(x_videos)
        y_videos = np.array(y_videos)
        # self.len = len(y)
        print('VOT dataset parsing done.')
        print('Total number of annotations in VOT dataset = %d' % (num_anno))
        print('Total number of videos in VOT dataset = %d' % (len(x_videos)))
        self.num_samples = 0
        for i in range(len(x_videos)):
            for j in range(len(x_videos[i])):
                self.index_dict[self.num_samples] = (i, j)
                self.video_dict[(i, j)] = self.num_samples
                self.num_samples += 1

        return x_videos, y_videos, z_videos, inp_ids, out_ids

    def get_quad(self, ann):
        """
        Parses ALOV annotation and returns bounding box in the format:
        [left, upper, width, height]
        """
        p_ = ann.strip().split(',')
        xbl = (float(p_[0]))
        ybl = (float(p_[1]))
        x1 = (float(p_[2]))
        y1 = (float(p_[3]))
        xtr = (float(p_[4]))
        ytr = (float(p_[5]))
        x2 = (float(p_[6]))
        y2 = (float(p_[7]))
        return np.array([xbl, ybl, x1, y1, xtr, ytr, x2, y2])

    def get_num_videos(self):
        return self.num_videos



if __name__ == '__main__':
    vot_root_dir = '../../data/VOT/'
    alov_root_dir = '../../data/ALOV/'
    vot = VotDataset(os.path.join(vot_root_dir,
                       'VOT_images/'),
                       os.path.join(vot_root_dir,
                       'VOT_ann/'),
                        os.path.join(vot_root_dir,
                       'VOT_results/'),)

    alov = AlovDataset(os.path.join(alov_root_dir,
                       'ALOV_images/'),
                       os.path.join(alov_root_dir,
                       'ALOV_ann/'),
                        os.path.join(alov_root_dir,
                        'ALOV_results/'),)

    # print(alov.get_orig_sample)
    iterations = 25
    epsilon = 0.05
    mode = 4
    N = 50
    start = 0
    end = N

    # model = pure_lkt(iterations, epsilon, mode)
    # while(end < alov.num_videos):
    #     # if(i % 20 == 0):
    #     #     print(i)
    
    #     data_x, data_y = get_data(alov, start, end)
    #     model.fit(data_x, data_y, batch_size=1, epochs=5)
    #     start = start + N
    #     end = min(end + N, alov.len)
    