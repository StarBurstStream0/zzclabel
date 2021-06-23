######################################@@@@@@@@@@@@@@#####
### TODO: for analysing all parsed anns
### DATE: 20210508
### AUTHOR: zzc

import os
import numpy as np
import matplotlib.pyplot as plt 
from zzclog import *
log = ZZCLOG(3)

def create_class_map(ann_dict):
    class_map = {}
    classes = []
    i = 0
    for ann in ann_dict:
        for obj in ann['object']:
            if obj['name'] not in classes:
                class_map[i] = obj['name']
                i += 1
            classes.append(obj['name'])
    cls_num = np.zeros(len(class_map))
    for i, class_ in class_map.items():
        cls_num[i] = classes.count(class_)
    return class_map, cls_num

def draw_data_distr(ann_dict, save_dir, save=True):
    class_map, cls_num = create_class_map(ann_dict)
    idxes = np.argsort(cls_num)
    classes = []
    for idx in idxes:
        classes.append(class_map[idx])
    log.info('class_map: ', class_map)
    
    plt.figure(figsize=(20,10))
    plt.bar(range(len(cls_num)), np.sort(cls_num))  
    for a,b in zip(range(len(cls_num)), np.sort(cls_num)): 
    # plt.bar(range(len(cls_num)), cls_num) 
    # for a,b in zip(range(len(cls_num)), cls_num): 
        plt.text(a, b+0.02, '%.0f' % b, ha='center', va= 'bottom', fontsize=16)
    plt.xticks(np.arange(len(cls_num)), classes, fontsize=16)
    plt.title('Data distribution by category', fontsize=16)
    plt.ylabel('number', fontsize=16)
    plt.xlabel('categories', fontsize=16)
    if save:
        plt.savefig(save_dir)
    # plt.show()  

# scale_level = [0, 1024, 9216]

def create_scale_map(ann_dict, scale_level):
    scale_map = np.zeros(3)
    for ann in ann_dict:
        for obj in ann['object']:
            area = float(obj['area'])
            if area <= scale_level[0]:
                log.error('{} has a wrong label! Please check.'.format(ann['image']['filename']))
            elif area > scale_level[0] and area <= scale_level[1]:
                scale_map[0] += 1
            elif area > scale_level[1] and area <= scale_level[2]:
                scale_map[1] += 1
            elif area > scale_level[2]:
                scale_map[2] += 1
    return list(scale_map)
                


def draw_data_pie(ann_dict, save_dir, scale_level, save=True):

    label = ['small: ['+str(scale_level[0])+'-'+str(scale_level[1])+']','mdeium: ['+str(scale_level[1])+'-'+str(scale_level[2])+']','large: ['+str(scale_level[2])+'-]']  
    scale_level = [x*x for x in scale_level] 
    explode =[0.05,0.05,0.05]
    color=['#FF69B4','#6495ED','#9ACD32']

    scale_map = create_scale_map(ann_dict, scale_level)

    log.debug('scale_map: ', scale_map)
    
    plt.figure(figsize=(20,10))
    plt.title('Data distribution by scale', fontsize=16)
    plt.pie(scale_map, labels=label,autopct='%.2f%%',explode=explode,
            colors=color,shadow=True)
    plt.legend(loc=4)
    if save:
        plt.savefig(save_dir)
    # plt.show()  

### TODO: for confusion matrix creation

import xml.etree.ElementTree as ET
import os
os.sys.path.append('/home/zzc/All/research/AerialDetection/DOTA_devkit')
#import cPickle
import numpy as np
import matplotlib.pyplot as plt
import polyiou
from functools import partial
import pdb
import shapely.geometry as shgeo
import sys
import json as js

def filter_points(points):
    if len(points) == 4:
        return points
    elif len(points) == 5:
        return points[:4]


def parse_gt(objs):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    for obj in objs:
        object_struct = {}
        object_struct['bbox'] = obj['segmentation'][0][:8]
        object_struct['name'] = obj['name']
        object_struct['difficult'] = 0
        objects.append(object_struct)

    # for shape in json['shapes']:
    #     object_struct = {}
    #     polygon = shape['points']
    #     points = np.array(polygon).astype(int)
    #     xmin, xmax, ymin, ymax = min(points[:, 0]), max(points[:, 0]), min(points[:, 1]), max(points[:, 1])
    #     label = shape['label']
    #     if xmax <= xmin:
    #         pass
    #     elif ymax <= ymin:
    #         pass
    #     else:

    #         # width, height = xmax - xmin, ymax - ymin
    #         # gtpoly = shgeo.Polygon(cal_area_bbox)
    #         # area = gtpoly.area
    #         # if area <= 80 or max(width, height) < 12:
    #         #     continue
    #         # ok!
    #         points = filter_points(points)
    #         points = list(points.reshape(-1))
    #         points = [int(point) for point in points]
    #         # print('points: ', points)
    #         object_struct['bbox'] = points
    #         object_struct['name'] = label
    #         object_struct['difficult'] = 0
    #         objects.append(object_struct)

    return objects

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(pred_dict,
             gt_dict,
             imagesetfile,
             classname_gt,
             classname_pred,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt

    # with open(imagesetfile, 'r') as f:
    #     lines = f.readlines()
    # imagenames = [x.strip() for x in lines]

    ########################################################################
    imagenames = []
    with open(imagesetfile, 'r') as f:
        json = js.load(f)
    for image in json['images']:
        image_name = image['file_name']
        if image_name not in imagenames:
            imagenames.append(image_name)
    ########################################################################

    recs = {}
    for i, imagename in enumerate(imagenames):
        #print('parse_files name: ', annopath.format(imagename))
        for ann in gt_dict:
            if ann['image']['filename'] == imagename:
                recs[imagename] = parse_gt(ann['object'])

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname_gt]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        # print('imagename: ', imagename)
        # print('bbox: ', bbox)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    ######################################################################
    # read dets from Task1* files
    # detfile = detpath.format(classname_pred)
    # with open(detfile, 'r') as f:
    #     lines = f.readlines()

    # splitlines = [x.strip().split(' ') for x in lines]
    # image_ids = [x[0] for x in splitlines]
    # confidence = np.array([float(x[1]) for x in splitlines])

    # print('check npred: ', len(confidence))

    # BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    image_ids = []
    confidence = []
    BB = []
    for ann in pred_dict:
        for obj in ann['object']:
            if obj['name'] != classname_pred:
                continue
            image_ids.append(ann['image']['filename'])
            confidence.append(obj['confidence'])
            BB.append(obj['segmentation'][0])
    confidence = np.array(confidence)
    BB = np.array(BB)

    ######################################################################

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    thresh = 0.2
    inds = np.sum(sorted_scores < -thresh)
    sorted_ind = sorted_ind[:inds]

    # print('check sorted_scores: ', sorted_scores)
    # print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    if len(BB) == 0:
        # print('BB: ', BB)
        return 0
    BB = BB[sorted_ind, :]


    image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        # R = class_recs[image_ids[d][:-4]]
        # print('image_ids[d]: ', image_ids[d])
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        # print('R[bbox]: ', R['bbox'])
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)


    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    # rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    # prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # ap = voc_ap(rec, prec, use_07_metric)

    # print('prec:', np.mean(prec))
    # print('rec:', np.mean(rec))

    # return rec, prec, ap
    # print(tp)
    if len(tp) > 0:
        return tp[-1]
    else:
        return 0

def plot_Matrix(cm, classes, title=None,  cmap=plt.cm.Blues):
    plt.rc('font',family='Times New Roman',size='16')   # 设置字体样式、大小
    
    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) == 0:
                cm[i, j]=0

    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j]*100 + 0.5) , fmt) + '%',
                        ha="center", va="center",
                        color="white"  if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cm.jpg', dpi=300)
