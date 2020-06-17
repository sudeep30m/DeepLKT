from os.path import join
from os import listdir
import json
import glob
import xml.etree.ElementTree as ET
from deeplkt.datasets.dataset import LKTDataset
import pickle as pkl
from deeplkt.utils.util import pkl_load, pkl_save
import numpy as np


class ImageNetDataset(LKTDataset):

    def __init__(self, root_dir, annot_dir, result_dir):
        self.index_dict = {}
        self.video_dict = {}

        self.x, self.y, self.inp_ids, self.out_ids = self._parse_data(root_dir, annot_dir, result_dir)
        self.num_videos = len(self.x)

    def _parse_data(self, root_dir, anno_dir, result_dir):

        x_videos = []
        y_videos = []
        inp_ids = []
        out_ids = []
        num_anno = 0

        for subset in listdir(anno_dir):
            subset_path = join(anno_dir, subset)
            videos = sorted(listdir(subset_path))
            for vi, video in enumerate(videos):
                if vi > 5:
                    break
                x = []
                y = []
                print('subset: {} video id: {:04d} / {:04d}'.format(subset, vi, len(videos)))
                video_path = join(subset_path, video)
                img_video_path = join(root_dir, subset, video)
                res_video_path = join(result_dir, subset, video)
                self.make_dir(res_video_path)
                self.make_dir(join(res_video_path, "images"))

                xmls = sorted(glob.glob(join(video_path, '*.xml')))
                trackids = set()
                frames = []
                for xml in xmls:
                    xmltree = ET.parse(xml)
                    objects = xmltree.findall('object')
                    objs = {}
                    for object_iter in objects:
                        trackid = int(object_iter.find('trackid').text)
                        trackids.add(trackid)
                        bndbox = object_iter.find('bndbox')

                        xmin, ymin, xmax, ymax = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                                        int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                        bndbox = [xmin, ymax, xmin, ymin, xmax, ymin, xmax, ymax]

                        objs[trackid] = bndbox
                    pth = join(img_video_path, xml.split('/')[-1].replace('xml', 'JPEG'))
                    objs['img_path'] = pth
                    frames.append(objs)
                for tid in trackids:
                    for i in range(len(frames) - 1):
                        if tid in frames[i] and tid in frames[i + 1]:
                            x.append((frames[i]['img_path'], frames[i + 1]['img_path']))
                            y.append((frames[i][tid], frames[i + 1][tid]))
                            num_anno += 1
                x_videos.append(x)
                y_videos.append(y)
                inp_ids.append(img_video_path)
                out_ids.append(res_video_path)
                
        print('ImageNet dataset parsing done.')
        print('Total number of annotations in ImageNet dataset = %d' % (num_anno))
        print('Total number of videos in ImageNet dataset = %d' % (len(x_videos)))

        self.num_samples = 0
        for i in range(len(x_videos)):
            for j in range(len(x_videos[i])):
                self.index_dict[self.num_samples] = (i, j)
                self.video_dict[(i, j)] = self.num_samples
                self.num_samples += 1

        return x_videos, y_videos, inp_ids, out_ids
    
    def get_quad(self, ann):
        ann = np.array([float(x) for x in ann])
        return ann


if __name__ == '__main__':
    # VID_base_path = '../../data/IMAGENET/ILSVRC2015'
    # ann_base_path = join(VID_base_path, 'Annotations/VID/train/')
    # img_base_path = join(VID_base_path, 'Data/VID/train/')
    # result_base_path = join(VID_base_path, 'Results/VID/train/')
    # imagenet = ImageNetDataset(img_base_path, ann_base_path, result_base_path)
    # pkl_save('imagenet.pkl', imagenet)
    imagenet = pkl_load('imagenet.pkl')
    imagenet.get_resized_image_with_gt(0, 1)
    # from IPython import embed; embed()
    