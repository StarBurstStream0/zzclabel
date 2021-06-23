######################################@@@@@@@@@@@@@@#####
### TODO: for visualizing all parsed anns
### DATE: 20210426
### AUTHOR: zzc

import os
import cv2
import numpy as np
from tqdm import tqdm
import time
from zzclog import *
log = ZZCLOG(2)

# tqdm parameters
desc = 'Visualized images'
mininterval = 0.2
ncols = 100

colors = [(255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (255, 0, 255),
          (0, 255, 255),
          (127, 0, 0),
          (0, 127, 0),
          (0, 0, 127),
          (127, 127, 0),
          (127, 0, 127),
          (0, 127, 127),
          (48, 128, 20),
          (255, 165, 0),
          (255, 182, 193),
          (255, 255, 255)]

def create_color_map(ann_dict):
    categories = []
    for ann in ann_dict:
        for obj in ann['object']:
            categories.append(obj['name'])
    categories = list(set(categories))
    color_map = {}
    for i, category in enumerate(categories):
        color_map[category] = colors[i]
    return color_map

def plot_seg(obj, image, color_map, thresh=0.5):
    confidence = obj['confidence']
    if isinstance(confidence, float):
        if confidence < thresh:
            return image
    label = obj['name']
    seg = obj['segmentation']
    seg = np.array(seg[0])
    # log.warning('seg: ', seg)
    if len(seg) % 2 == 1:
        seg = seg[:-1]
    seg = seg.reshape(int(len(seg) / 2), 2)
    for j in range(len(seg)):
        cv2.line(image, (int(seg[j, 0]), int(seg[j, 1])), (int(seg[(j+1)%len(seg), 0]), int(seg[(j+1)%len(seg), 1])), color_map[label], 2)
    # log.debug('label: ', label)
    # log.debug('confidence: ', confidence)
    # log.debug('int(seg[0, 0]): ', int(seg[0, 0]))
    # log.debug('color_map[label]: ', color_map[label])
    if isinstance(confidence, float):
        cv2.putText(image, '%s %.3f' % (label, confidence), (int(seg[0, 0]), int(seg[0, 1]) + 10), color=color_map[label], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    elif isinstance(confidence, str):
        cv2.putText(image, '%s' % (label), (int(seg[0, 0]), int(seg[0, 1]) + 10), color=color_map[label], fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return image

def plot_mask(obj, image, color_map, thresh=0.5):
    label = obj['name']
    mask = obj['mask'][0].astype(bool)
    image[mask] = image[mask] * 0.5 + np.array(color_map[label]) * 0.5
    return image

def vis_labels(ann_dict, img_dir, save_dir, type='segm', num=-1, thresh=0.5):
    log.info('Start to visualize images with anns...')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        log.info('Created saving dir {}...'.format(save_dir))
    color_map = create_color_map(ann_dict)
    log.debug('color_map: ', color_map)
    log.info('Successfully created color map...')
    if num <= 0:
        num = len(ann_dict)
    else:
        num = num
    time.sleep(0.5)
    img_name = ''
    for i, ann in enumerate(tqdm(ann_dict[:num], desc = desc, mininterval = mininterval, ncols = ncols)):
        if img_name != ann['image']['filename'].split('/')[-1]:
            img_name = ann['image']['filename'].split('/')[-1]
            log.debug('img_dir: ', img_dir)
            log.debug('img_name: ', img_name)
            image = cv2.imread(os.path.join(img_dir, img_name))
            log.debug('image: ', image.shape)
        for obj in ann['object']:
            # label = obj['name']
            if isinstance(type, list):
                if 'segm' in type:
                    image = plot_seg(obj, image, color_map, thresh)
                if 'bbox' in type:
                    pass
                if 'mask' in type:
                    image = plot_mask(obj, image, color_map, thresh)
            elif isinstance(type, str):
                if type == 'segm':
                    image = plot_seg(obj, image, color_map, thresh)
                elif type == 'bbox':
                    pass
                elif type == 'mask':
                    image = plot_mask(obj, image, color_map, thresh)

        log.debug('Saving images...')
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, image)
    time.sleep(0.5)
    log.info('Finish all images visualization!')
        