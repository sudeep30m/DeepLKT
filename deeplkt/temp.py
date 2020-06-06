from deeplkt.datasets.dataset import *
from deeplkt.utils.util import pkl_load, pkl_save
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def plot_bar_graph(results, path):
    cmap = plt.get_cmap('tab10')
    for k in results:
        num = len(results[k])
    # print(num)
    results['x'] = np.array(range(1, num + 1))
    colors = [cmap(i) for i in np.linspace(0, 1, len(results))]

    df=pd.DataFrame(results)
    df['sort_val'] = df.grey_pure_lkt - df.grey_learned_lkt
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

use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda") if use_cuda else torch.device("cpu")
vot_root_dir = '../../data/VOT/'

vot = VotDataset(os.path.join(vot_root_dir,
                       'VOT_images/'),
                 os.path.join(vot_root_dir,
                       'VOT_ann/'),
                 os.path.join(vot_root_dir,
                       'VOT_results/'), 
                 device)

results = pkl_load('grey-synth-results-pair.pkl')
pure_lkt = results['grey_pure_lkt']
learned_lkt = results['grey_learned_lkt']

# print(pure_lkt[19], learned_lkt[19])
total = 0
pure_lkt_iou = 0
vgg_lkt_iou = 0


for i in range(25):
    pure_lkt_iou += pure_lkt[i] * vot.get_num_images(i) 
    vgg_lkt_iou += learned_lkt[i] * vot.get_num_images(i) 
    total += vot.get_num_images(i)
pure_lkt_iou /= total
vgg_lkt_iou /= total
# print("E1 pure")
print(pure_lkt_iou, vgg_lkt_iou)
# p
plot_bar_graph(results, "grey-synth-results-pair.png")





