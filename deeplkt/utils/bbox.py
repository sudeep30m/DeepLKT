# Copyright (c) SenseTime. All Rights Reserved.

from collections import namedtuple

import numpy as np
import torch

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
# alias
BBox = Corner
Center = namedtuple('Center', 'x y w h')


def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x1, y1, x2, y2 = corner
        return Center((x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1), (y2 - y1))
    else:
        x1, y1, x2, y2 = corner[0], corner[1], corner[2], corner[3]
        x = (x1 + x2) * 0.5
        y = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        return x, y, w, h


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        x, y, w, h = center
        return Corner(x - w * 0.5, y - h * 0.5, x + w * 0.5, y + h * 0.5)
    else:
        x, y, w, h = center[0], center[1], center[2], center[3]
        x1 = x - w * 0.5
        y1 = y - h * 0.5
        x2 = x + w * 0.5
        y2 = y + h * 0.5
        return x1, y1, x2, y2


def IoU(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (x1, y1, x2, y2)
        rect2: (x1, y1, x2, y2)
    Returns:
        iou
    """
    # overlap
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)
    target_a = (tx2-tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou


def cxy_wh_2_rect(pos):
    """ convert (cx, cy, w, h) to (x1, y1, w, h), 0-index
    """
    return np.array([pos[:, 0]-pos[:, 2]/2, pos[:, 1]-pos[:, 3]/2, pos[:, 2], pos[:, 3]]).transpose()


def rect_2_cxy_wh(rect):
    """ convert (x1, y1, w, h) to (cx, cy, w, h), 0-index
    """
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), \
        np.array([rect[2], rect[3]])


def cxy_wh_2_rect1(pos, sz):
    """ convert (cx, cy, w, h) to (x1, y1, w, h), 1-index
    """
    return np.array([pos[0]-sz[0]/2+1, pos[1]-sz[1]/2+1, sz[0], sz[1]])


def rect1_2_cxy_wh(rect):
    """ convert (x1, y1, w, h) to (cx, cy, w, h), 1-index
    """
    return np.array([rect[0]+rect[2]/2-1, rect[1]+rect[3]/2-1]), \
        np.array([rect[2], rect[3]])



def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = region.size
    if nv == 8:
        cx = np.mean(region[0::2])
        cy = np.mean(region[1::2])
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return cx, cy, w, h


def get_min_max_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by min-max box
    """
    # print(region.shape)
    # nv = region.size
    # if nv == 8:
    cx = np.mean(region[:, 0::2], 1)
    cy = np.mean(region[:, 1::2], 1)
    x1 = np.min(region[:, 0::2], 1)
    x2 = np.max(region[:, 0::2], 1)
    y1 = np.min(region[:, 1::2], 1)
    y2 = np.max(region[:, 1::2], 1)
    # print(cx.shape, cy.shape, x1.shape, x2.shape, y1.shape, y2.shape)
    w = x2 - x1
    h = y2 - y1
    # else:
    #     x = region[0]
    #     y = region[1]
    #     w = region[2]
    #     h = region[3]
    #     cx = x+w/2
    #     cy = y+h/2
    return np.array([cx, cy, w, h]).transpose()

def get_min_max_bbox_torch(region):
    """ convert region to (cx, cy, w, h) that represent by min-max box
    """
    nv = region.size()[0]
    # print("&&&&&&&&&", region, nv)
    if nv == 8:
        cx = torch.mean(region[0::2])
        cy = torch.mean(region[1::2])
        x1 = torch.min(region[0::2])
        x2 = torch.max(region[0::2])
        y1 = torch.min(region[1::2])
        y2 = torch.max(region[1::2])
        w = x2 - x1 
        h = y2 - y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x+w/2
        cy = y+h/2
    return np.array([cx, cy, w, h]).transpose()

def get_region_from_center(rect):
    """ convert (x1, y1, w, h) to region
    """
    
    x1 = rect[:, 0] - rect[:, 2] / 2.0
    y1 = rect[:, 1] - rect[:, 3] / 2.0
    x2 = rect[:, 0] + rect[:, 2] / 2.0
    y2 = rect[:, 1] + rect[:, 3] / 2.0

    return np.array([x1, y2, x1, y1, x2, y1, x2, y2]).transpose() 

def get_region_from_corner(rect):
    """ convert (x1, y1, w, h) to region
    """
    
    rect[:, 2] += rect[:, 0]
    rect[:, 3] += rect[:, 1]
    return np.array([rect[:, 0], rect[:, 3], rect[:, 0], rect[:, 1], rect[:, 2], 
                    rect[:, 1], rect[:, 2], rect[:, 3]]).transpose()