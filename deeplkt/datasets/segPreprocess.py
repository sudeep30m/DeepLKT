from PIL import Image
import cv2
from torchvision import transforms
import numpy as np
import torch

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

class Preprocess:

    def __init__(self, opt):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        
        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


    def get_images(self, patches):
        img_resized_list = [[] for x in range(len(self.imgSizes))]

        for im_patch in patches:
            im_patch = im_patch[0].transpose(1, 2, 0)
            img = im_patch.astype(np.uint8)
            # print(type(img))
            # print(img.shape)
            img = Image.fromarray( cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img.save("img.jpeg")
            ori_width, ori_height = img.size
            
            for j, this_short_size in enumerate(self.imgSizes):
                # calculate target height and width
                scale = min(this_short_size / float(min(ori_height, ori_width)),
                            self.imgMaxSize / float(max(ori_height, ori_width)))
                target_height, target_width = int(ori_height * scale), \
                                              int(ori_width * scale)

                # to avoid rounding in network
                target_width = self.round2nearest_multiple(target_width, \
                                                    self.padding_constant)
                target_height = self.round2nearest_multiple(target_height, \
                                                    self.padding_constant)

                # resize images
                img_resized = imresize(img, (target_width, target_height),\
                                        interp='bilinear')
                img_resized.save(str(target_height) + ".jpeg")
                img_resized = self.img_transform(img_resized)
                img_resized = torch.unsqueeze(img_resized, 0)
                # print(img_resized.shape)
                img_resized_list[j].append(img_resized)
        # from IPython import embed;embed()
        img_resized_list = [torch.cat(x).contiguous() \
                            for x in img_resized_list]
        return img_resized_list

