######################################@@@@@@@@@@@@@@#####
### TODO: for several useful tools
### DATE: 20210517
### AUTHOR: zzc
import random
import copy
from template import *
import numpy as np

def cal_area(obj):
    xmin = obj['bndbox']['xmin'] 
    xmax = obj['bndbox']['xmax'] 
    ymin = obj['bndbox']['ymin'] 
    ymax = obj['bndbox']['ymax'] 
    obj['area'] = int((xmax-xmin) * (ymax-ymin))
    return obj

def split_dataset(ann_dict, ratio=0.7):
    train_ann_dict = []    
    test_ann_dict = []
    for ann in ann_dict:
        if random.random() <= ratio:
            train_ann_dict.append(ann)
        else:
            test_ann_dict.append(ann)
    return train_ann_dict, test_ann_dict

### TODO: for class-agnostic nms ops

def obb2hbb(obb):
    xmin = np.min(obb[0::2])
    xmax = np.max(obb[0::2])
    ymin = np.min(obb[1::2])
    ymax = np.max(obb[1::2])
    return np.array([xmin, ymin, xmax, ymax])

def cal_iou(obb1, obb2):
    hbb1 = obb2hbb(obb1)    
    hbb2 = obb2hbb(obb2)
#     print('hbb1: ', hbb1)
#     print('hbb2: ', hbb2)
    hbbs = np.stack((hbb1, hbb2))
    inter = (np.min(hbbs[:,2]) - np.max(hbbs[:,0])) * (np.min(hbbs[:,3]) - np.max(hbbs[:,1]))
    union = (np.max(hbbs[:,2]) - np.min(hbbs[:,0])) * (np.max(hbbs[:,3]) - np.min(hbbs[:,1]))
    return inter / union

def delete_el(obbs, scores, classes, deleted_idx):
    new_obbs = []
    new_scores = []
    new_classes = []
    assert len(obbs) == len(scores)
    for i in range(len(obbs)):
        if i in deleted_idx:
            continue
        else:
            new_obbs.append(obbs[i])
            new_scores.append(scores[i])
            new_classes.append(classes[i])
    return new_obbs, new_scores, new_classes

def classes_nms_per_image(obbs, scores, classes, thresh=0.5):
    deleted_idx = []
    for i in range(len(obbs)-1):
        obb = obbs[i]
        for j in range(i+1, len(obbs)):
            if j in deleted_idx:
                continue
            iou = cal_iou(obb, obbs[j])
            if iou >= thresh:
                deleted_idx.append(j)
                # print('{} and {} iou not allowed! is {}'.format(obb, obbs[j], iou))

    new_obbs, new_scores, new_classes = delete_el(obbs, scores, classes, deleted_idx)
    return new_obbs, new_scores, new_classes

def classes_nms(ann_dict, thresh=0.5):
    images_list = []
    for ann in ann_dict:
        images_list.append(ann['image']['filename'])
    images_list = list(set(images_list))
    
    nms_ann_dict = []
    for image in images_list:
        obbs = []
        scores = []
        classes = []
        new_ann = copy.deepcopy(ann_template)
        for ann in ann_dict:
            if ann['image']['filename'] == image:
                new_ann['image'] = ann['image']
                for obj in ann['object']:
                    obbs.append(obj['segmentation'][0])
                    scores.append(obj['confidence'])
                    classes.append(obj['name'])
        obbs = np.array(obbs)
        scores = np.array(scores)

        idx = np.argsort(-scores)
        scores = scores[idx]
        obbs = obbs[idx]
        classes_ = []
        for i in idx:
            classes_.append(classes[i])
        new_obbs, new_scores, new_classes = classes_nms_per_image(obbs, scores, classes_, thresh=thresh)
        for nobb, bscore, bclass in zip(new_obbs, new_scores, new_classes):
            obj= copy.deepcopy(obj_template)
            obj['name'] = bclass
            obj['segmentation'].append(nobb)
            obj['confidence'] = bscore
            new_ann['object'].append(obj)
        nms_ann_dict.append(new_ann)
    return nms_ann_dict