######################################@@@@@@@@@@@@@@#####
### TODO: for reading and parsing all types of anns
### DATE: 20210422
### AUTHOR: zzc

import os
import json as js
from template import *
from utils import *
import copy
from collections import OrderedDict
import xmltodict
import numpy as np
import math
from zzclog import *
from tqdm import tqdm
import time
import pickle
import cv2

# LEVEL = 3
# for zzclog 0.2.3 version:
log = ZZCLOG(2)

# tqdm parameters
desc = 'Loaded annotations'
mininterval = 0.2
ncols = 100


def create_class_map_json(json_dict):
    class_map = {}
    for class_ in json_dict:
        class_map[class_['id']] = class_['name']
    return class_map

def check_path(path, format='json'):
    # input path must be abs path!
    # format: json, xml, dir
    if not os.path.exists(path):
        log.error('Input file must exist! {}'.format(path))
        return 0
    if format == 'json':
        if path[-4:] != 'json':
            log.error('Input file must be a json format! {}'.format(path.split('/')[-1]))
            return 0
        elif path.split('/')[-1][0] == '.':
            log.error('Input file must be a json format! {}'.format(path.split('/')[-1]))
            return 0
    elif format == 'xml':
        if path[-3:] != 'xml':
            log.error('Input file must be a xml format! {}'.format(path.split('/')[-1]))
            return 0
        elif path.split('/')[-1][0] == '.':
            log.error('Input file must be a xml format! {}'.format(path.split('/')[-1]))
            return 0
    elif format == 'pkl':
        if path[-3:] != 'pkl':
            log.error('Input file must be a pkl format! {}'.format(path.split('/')[-1]))
            return 0
        elif path.split('/')[-1][0] == '.':
            log.error('Input file must be a pkl format! {}'.format(path.split('/')[-1]))
            return 0
    elif format == 'txt':
        if path[-3:] != 'txt':
            log.error('Input file must be a txt format! {}'.format(path.split('/')[-1]))
            return 0
        elif path.split('/')[-1][0] == '.':
            log.error('Input file must be a txt format! {}'.format(path.split('/')[-1]))
            return 0
    elif format == 'png':
        if path[-3:] != 'png':
            log.error('Input file must be a png format! {}'.format(path.split('/')[-1]))
            return 0
        elif path.split('/')[-1][0] == '.':
            log.error('Input file must be a png format! {}'.format(path.split('/')[-1]))
            return 0
    elif format == 'dir':
        pass
    return 1

def read_coco(json_path):
    if not check_path(json_path, format='json'):
        return None
    with open(json_path, 'r') as f:
        json = js.load(f)
    class_map = create_class_map_json(json['categories'])
    ann_dict = []
    # try:
    #     with tqdm(json['images'], position=0) as t:
    #         for file_name in t:
    for file_name in tqdm(json['images'], desc = desc, mininterval = mininterval, ncols = ncols):
        ann = copy.deepcopy(ann_template)

        ann['image']['filename'] = file_name['file_name']
        ann['image']['size']['width'] = file_name['width']
        ann['image']['size']['height'] = file_name['height']

        ###############################################
        ### TODO: check if image exists
        # image_path = os.path.join(images_path.format(type), file_name['file_name'])
        # if os.path.exists(image_path):
        #     image = cv2.imread(image_path)
        #     h, w, c = image.shape
        #     xml_dict['annotation']['size']['width'] = w
        #     xml_dict['annotation']['size']['height'] = h
        #     xml_dict['annotation']['size']['depth'] = c
        # else:
        #     print('Wrong! image {} not exists!!!'.format(image_path))
        ###############################################

        for i in range(len(json['annotations'])):
            if json['annotations'][i]['image_id'] == file_name['id']:
                bbox = json['annotations'][i]['bbox']
                points = np.array(bbox)
                xmin, xmax, ymin, ymax = points[0], points[0] + points[2], points[1], points[1] + points[3]
                label = json['annotations'][i]['category_id']
                if xmax <= xmin:
                    continue
                elif ymax <= ymin:
                    continue
                else:
                    obj= copy.deepcopy(obj_template)
                    obj['name'] = class_map[label]
                    obj['bndbox']['xmin'] = xmin
                    obj['bndbox']['xmax'] = xmax
                    obj['bndbox']['ymin'] = ymin
                    obj['bndbox']['ymax'] = ymax
                    obj['area'] = (ymax - ymin) * (xmax - xmin)
                    obj['segmentation'].append(json['annotations'][i]['segmentation'][0])
                    ann['object'].append(obj)
        ann_dict.append(ann)
        # time.sleep(0.1)
    # except KeyboardInterrupt:
    #     t.close()
    #     raise
    # t.close()
    time.sleep(0.1)
    log.info('class_map: ', class_map)

    return ann_dict

def read_labelme(json_path):
    if not check_path(json_path, format='dir'):
        return None
    ann_dict = []
    # log.debug(json_path)
    for json_file in tqdm(os.listdir(json_path), desc = desc, mininterval = mininterval, ncols = ncols):
        json_name = os.path.join(json_path, json_file)
        if not check_path(json_name, format='json'):
            continue
        # log.debug(json_name)
        with open(json_name, 'r', encoding='utf-8') as f:
            json = js.load(f)
        
        ann= copy.deepcopy(ann_template)
        ann['image']['filename'] = json['imagePath'].split('/')[-1]
        ann['image']['size']['width'] = json['imageWidth']
        ann['image']['size']['height'] = json['imageHeight']

        for shape in json['shapes']:
            polygon = shape['points']
            points = np.array(polygon).astype(int)
            xmin, xmax, ymin, ymax = min(points[:, 0]), max(points[:, 0]), min(points[:, 1]), max(points[:, 1])
            label = shape['label']
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                # obj_dict = obj_template.copy()
                obj= copy.deepcopy(obj_template)
                obj['name'] = label
                obj['bndbox']['xmin'] = xmin
                obj['bndbox']['xmax'] = xmax
                obj['bndbox']['ymin'] = ymin
                obj['bndbox']['ymax'] = ymax
                obj = cal_area(obj)
                points = list(points.reshape(-1))
                points = [int(point) for point in points]
                obj['segmentation'].append(points)

                ann['object'].append(obj)
        ann_dict.append(ann)
    return ann_dict



def read_json(json_path, format='COCO'):
    log.info('Start to load the json annotations...')
    time.sleep(0.5)
    # format: COCO, labelme
    ann_dict = None
    if format == 'COCO':
        ann_dict = read_coco(json_path)
    elif format == 'labelme':
        ann_dict = read_labelme(json_path)
    time.sleep(0.5)
    if ann_dict is not None:
        log.info('Finish loading the json annotations!')
    else:
        log.warning('Finish loading the json annotations, but get no result...')
    return ann_dict

def five2eight(obb):
    # angle = math.degrees(obb[4])
    # vh = math.sin(angle) * obb[3]
    # vw = math.cos(angle) * obb[3]
    # x1 = obb[0] + vw/2
    # x3 = obb[0] - vw/2
    # y1 = obb[1] + vh/2
    # y3 = obb[1] - vh/2

    # hh = math.sin(angle) * obb[2]
    # hw = math.cos(angle) * obb[2]
    # x4 = obb[0] + hw/2
    # x2 = obb[0] - hw/2
    # y4 = obb[1] + hh/2
    # y2 = obb[1] - hh/2

    cs = np.cos(obb[4])
    ss = np.sin(obb[4])
    w = obb[2] - 1
    h = obb[3] - 1

    ## change the order to be the initial definition
    x_ctr = obb[0]
    y_ctr = obb[1]
    x1 = x_ctr + cs * (w / 2.0) - ss * (-h / 2.0)
    x2 = x_ctr + cs * (w / 2.0) - ss * (h / 2.0)
    x3 = x_ctr + cs * (-w / 2.0) - ss * (h / 2.0)
    x4 = x_ctr + cs * (-w / 2.0) - ss * (-h / 2.0)

    y1 = y_ctr + ss * (w / 2.0) + cs * (-h / 2.0)
    y2 = y_ctr + ss * (w / 2.0) + cs * (h / 2.0)
    y3 = y_ctr + ss * (-w / 2.0) + cs * (h / 2.0)
    y4 = y_ctr + ss * (-w / 2.0) + cs * (-h / 2.0)

    return [int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)]

def read_HRSC(xml_dir):
    if not check_path(xml_dir, format='dir'):
        return None
    ann_dict = []
    for xml_name in tqdm(os.listdir(xml_dir), desc = desc, mininterval = mininterval, ncols = ncols):
        xml_path = os.path.join(xml_dir, xml_name)
        if not check_path(xml_path, format='xml'):
            continue
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_dict = xmltodict.parse(f.read())

        ann= copy.deepcopy(ann_template)
        ann['image']['filename'] = xml_dict['HRSC_Image']['Img_FileName']+'.'\
            +xml_dict['HRSC_Image']['Img_FileFmt']
        ann['image']['size']['width'] = int(xml_dict['HRSC_Image']['Img_SizeWidth'])
        ann['image']['size']['height'] = int(xml_dict['HRSC_Image']['Img_SizeHeight'])

        ### TODO: if a xml has no target, we should continue
        try:
            tmp = xml_dict['HRSC_Image']['HRSC_Objects']['HRSC_Object']
        except:
            continue
        ### TODO: if a xml has one object, it is not saved in list, but directly in a orderedict. 
        if not isinstance(xml_dict['HRSC_Image']['HRSC_Objects']['HRSC_Object'], list):
            HRSC_obj = xml_dict['HRSC_Image']['HRSC_Objects']['HRSC_Object']
            # log.debug('HRSC_obj: ', HRSC_obj)
            xmin, xmax, ymin, ymax = int(HRSC_obj['box_xmin']), int(HRSC_obj['box_xmax']),\
                int(HRSC_obj['box_ymin']), int(HRSC_obj['box_ymax'])
            label = HRSC_obj['Class_ID']
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                obj= copy.deepcopy(obj_template)
                # log.debug("xml_dict: ", xml_dict)
                obj['name'] = label
                obj['bndbox']['xmin'] = xmin
                obj['bndbox']['xmax'] = xmax
                obj['bndbox']['ymin'] = ymin
                obj['bndbox']['ymax'] = ymax
                cx, cy, w, h, a = float(HRSC_obj['mbox_cx']), float(HRSC_obj['mbox_cy']),\
                    float(HRSC_obj['mbox_w']), float(HRSC_obj['mbox_h']), float(HRSC_obj['mbox_ang'])
                obb = [cx, cy, w, h, a]
                rbb = five2eight(obb)
                obj['segmentation'].append(rbb)
                ann['object'].append(obj)

            ann_dict.append(ann)
        else:
            for HRSC_obj in xml_dict['HRSC_Image']['HRSC_Objects']['HRSC_Object']:
                # log.debug('HRSC_obj: ', HRSC_obj)
                xmin, xmax, ymin, ymax = int(HRSC_obj['box_xmin']), int(HRSC_obj['box_xmax']),\
                    int(HRSC_obj['box_ymin']), int(HRSC_obj['box_ymax'])
                label = HRSC_obj['Class_ID']
                if xmax <= xmin:
                    pass
                elif ymax <= ymin:
                    pass
                else:
                    obj= copy.deepcopy(obj_template)
                    # log.debug("xml_dict: ", xml_dict)
                    obj['name'] = label
                    obj['bndbox']['xmin'] = xmin
                    obj['bndbox']['xmax'] = xmax
                    obj['bndbox']['ymin'] = ymin
                    obj['bndbox']['ymax'] = ymax
                    
                    cx, cy, w, h, a = float(HRSC_obj['mbox_cx']), float(HRSC_obj['mbox_cy']),\
                        float(HRSC_obj['mbox_w']), float(HRSC_obj['mbox_h']), float(HRSC_obj['mbox_ang'])
                    obb = [cx, cy, w, h, a]
                    rbb = five2eight(obb)
                    obj['segmentation'].append(rbb)
                    ann['object'].append(obj)
            ann_dict.append(ann)
    return ann_dict

def read_voc(xml_dir):
    if not check_path(xml_dir, format='dir'):
        return None
    ann_dict = []
    for xml_name in tqdm(os.listdir(xml_dir), desc = desc, mininterval = mininterval, ncols = ncols):
        xml_path = os.path.join(xml_dir, xml_name)
        log.debug('xml_name: ', xml_name)
        if not check_path(xml_path, format='xml'):
            continue
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_dict = xmltodict.parse(f.read())

        ann= copy.deepcopy(ann_template)
        ann['image']['filename'] = xml_dict['annotation']['filename'].split('/')[-1]
        ann['image']['size']['width'] = int(xml_dict['annotation']['size']['width'])
        ann['image']['size']['height'] = int(xml_dict['annotation']['size']['height'])
        try:
            ann['image']['size']['depth'] = xml_dict['annotation']['size']['depth']
        except:
            pass

        if not isinstance(xml_dict['annotation']['object'], list):
            voc_obj = xml_dict['annotation']['object']
            # log.debug('HRSC_obj: ', HRSC_obj)
            xmin, xmax, ymin, ymax = int(voc_obj['bndbox']['xmin']), int(voc_obj['bndbox']['xmax']),\
                int(voc_obj['bndbox']['ymin']), int(voc_obj['bndbox']['ymax'])
            label = voc_obj['name']
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                obj= copy.deepcopy(obj_template)
                # log.debug("xml_dict: ", xml_dict)
                obj['name'] = label
                obj['bndbox']['xmin'] = xmin
                obj['bndbox']['xmax'] = xmax
                obj['bndbox']['ymin'] = ymin
                obj['bndbox']['ymax'] = ymax
                try:
                    seg = voc_obj['segmentation'][1:-1].replace(' ', '').split(',')
                    seg = [int(x) for x in seg]
                    obj['segmentation'].append(seg)
                except:
                    obb = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                    obj['segmentation'].append(obb)
                ann['object'].append(obj)

            ann_dict.append(ann)
        else:
            for voc_obj in xml_dict['annotation']['object']:
                # log.debug('HRSC_obj: ', HRSC_obj)
                xmin, xmax, ymin, ymax = int(voc_obj['bndbox']['xmin']), int(voc_obj['bndbox']['xmax']),\
                    int(voc_obj['bndbox']['ymin']), int(voc_obj['bndbox']['ymax'])
                label = voc_obj['name']
                if xmax <= xmin:
                    pass
                elif ymax <= ymin:
                    pass
                else:
                    obj= copy.deepcopy(obj_template)
                    # log.debug("xml_dict: ", xml_dict)
                    obj['name'] = label
                    obj['bndbox']['xmin'] = xmin
                    obj['bndbox']['xmax'] = xmax
                    obj['bndbox']['ymin'] = ymin
                    obj['bndbox']['ymax'] = ymax
                    try:
                        seg = voc_obj['segmentation'][1:-1].replace(' ', '').split(',')
                        seg = [int(x) for x in seg]
                        obj['segmentation'].append(seg)
                    except:
                        obb = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                        obj['segmentation'].append(obb)
                    ann['object'].append(obj)
            ann_dict.append(ann)
    return ann_dict

def read_xml(xml_dir, format='voc'):
    # format: coco, labelme
    log.info('Start to load the xml annotations...')
    time.sleep(0.5)
    ann_dict = None
    if format == 'voc':
        ann_dict = read_voc(xml_dir)
    elif format == 'HRSC':
        ann_dict = read_HRSC(xml_dir)
    time.sleep(0.5)
    if ann_dict is not None:
        log.info('Finish loading the xml annotations!')
    else:
        log.warning('Finish loading the xml annotations, but get no result...')
    return ann_dict

def read_pkl(pkl_path, json_path, with_segm=False):
    # format: pkl
    log.info('Start to load the pkl results...')
    if not check_path(pkl_path, format='pkl'):
        return None
    if not check_path(json_path, format='json'):
        return None
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    with open(json_path, 'r') as f:
        json = js.load(f)
    class_map = create_class_map_json(json['categories'])
    ann_dict = []
    #################################################################
    # TODO: find the bias between image ID and practical number
    bias = 1000000
    for file_name in json['images']:
        if int(file_name['id']) < bias:
            bias = int(file_name['id'])
    # log.warning('bias: ', bias)
    time.sleep(0.5)
    #################################################################
    for file_name in tqdm(json['images'], desc = desc, mininterval = mininterval, ncols = ncols):
        ann = copy.deepcopy(ann_template)

        ann['image']['filename'] = file_name['file_name']
        ann['image']['size']['width'] = file_name['width']
        ann['image']['size']['height'] = file_name['height']
        result = results[int(file_name['id'] - bias)]

        # for i in range(len(result)):
        #     for j in range(len(result[i])):
        #         bbox = result[i][j].tolist()
        for i in range(len(result[0])):
            for j in range(len(result[0][i])):
                bbox = result[0][i][j].tolist()
                points = np.array(bbox)
                confidence = points[-1]
                if len(bbox) <= 4:
                    bbox = [points[0], points[1], points[2], points[1], points[2], points[3], points[0], points[3]]
                xmin, xmax, ymin, ymax = points[0], points[2], points[1], points[3]
                label = i+1
                if xmax <= xmin:
                    continue
                elif ymax <= ymin:
                    continue
                else:
                    obj= copy.deepcopy(obj_template)
                    obj['name'] = class_map[label]
                    obj['bndbox']['xmin'] = xmin
                    obj['bndbox']['xmax'] = xmax
                    obj['bndbox']['ymin'] = ymin
                    obj['bndbox']['ymax'] = ymax
                    obj['segmentation'].append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
                    obj['confidence'] = confidence
                    ann['object'].append(obj)

                #########
                if with_segm == True:
                    import pycocotools.mask as mask_util
                    mask = mask_util.decode(result[1][i][j])  # encoded with RLE
                    obj['mask'].append(mask)
                #########
        ann_dict.append(ann)
        # time.sleep(0.1)
    # except KeyboardInterrupt:
    #     t.close()
    #     raise
    # t.close()
    log.info('class_map: ', class_map)
    time.sleep(0.1)
    return ann_dict

def create_class_map_txt_result(txt_dir):
    class_map = {}
    for i, filename in enumerate(os.listdir(txt_dir)):
        class_map[i] = filename.replace('.txt', '')
    return class_map

def read_txt_result(txt_dir, img_dir, ext='jpg'):
    # format: txt
    log.info('Start to load the txt results...')
    if not check_path(txt_dir, format='dir'):
        return None
    class_map = create_class_map_txt_result(txt_dir)
    log.info('class_map: ', class_map)
    time.sleep(0.5)
    ann_dict = []
    images_list = []
    for filename in os.listdir(txt_dir):
        label = filename.replace('.txt', '')
        file_path = os.path.join(txt_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            results = f.readlines()
        ann = copy.deepcopy(ann_template)
        for i, result in enumerate(tqdm(results, desc = label, mininterval = mininterval, ncols = ncols)):
            result = result.split(' ')
            log.debug('result: ', result)
            image_name = result[0] + '.' + ext

            if image_name not in images_list:
                image_path = os.path.join(img_dir, image_name)
                if not os.path.exists(image_path):
                    log.warning('{} gets no corresponding images!'.format(result[0]))
                    continue

                images_list.append(image_name)
                ann = copy.deepcopy(ann_template)
                ann['image']['filename'] = image_name
                image = cv2.imread(image_path)
                h, w = image.shape[0], image.shape[1]
                ann['image']['size']['width'] = w
                ann['image']['size']['height'] = h

            points = [float(result[j]) for j in range(2, len(result))]
            confidence = float(result[1])
            if len(points) == 4:
                xmin, xmax, ymin, ymax = points[0], points[2], points[1], points[3]
                points = [points[0], points[1], points[2], points[1], points[2], points[3], points[0], points[3]]
            else:
                xmin = min(points[0::2])
                xmax = max(points[0::2])
                ymin = min(points[1::2])
                ymax = max(points[1::2])
            if xmax <= xmin:
                continue
            elif ymax <= ymin:
                continue
            else:
                obj= copy.deepcopy(obj_template)
                obj['name'] = label
                obj['bndbox']['xmin'] = xmin
                obj['bndbox']['xmax'] = xmax
                obj['bndbox']['ymin'] = ymin
                obj['bndbox']['ymax'] = ymax
                obj['segmentation'].append(points)
                obj['confidence'] = confidence
                ann['object'].append(obj)
            if i == len(results)-1:
                ann_dict.append(ann)
            elif results[i+1].split(' ')[0] + '.' + ext not in images_list:
                ann_dict.append(ann)
    return ann_dict

def merge_anns(anns_list):
    ann = copy.deepcopy(anns_list[0])
    for i in range(1, len(anns_list)):
        ann['object'].append(anns_list[i]['object'][0])
    return ann

def filter_dict(ann_dict):
    new_dict = []
    images_list = []
    for ann in ann_dict:
        images_list.append(ann['image']['filename'])
    images_list = list(set(images_list))
    for image_name in images_list:
        anns_list = []
        for ann in ann_dict:
            if ann['image']['filename'] == image_name:
                anns_list.append(ann)
        new_dict.append(merge_anns(anns_list))
    return new_dict

def read_DOTA(txt_dir, img_dir, ext='png'):
    # format: DOTA origin txt
    log.info('Start to load the txt results...')
    if not check_path(txt_dir, format='dir'):
        return None
    class_map = create_class_map_txt_result(txt_dir)
    log.info('class_map: ', class_map)
    time.sleep(0.5)
    ann_dict = []
    
    for filename in os.listdir(txt_dir):
        images_list = []
        label = filename.replace('.txt', '')
        file_path = os.path.join(txt_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            results = f.readlines()
        # ann = copy.deepcopy(ann_template)
        for i, result in enumerate(tqdm(results, desc = label, mininterval = mininterval, ncols = ncols)):
            result = result.split(' ')
            log.debug('result: ', result)
            # image_name = result[0] + '.' + ext
            image_name = result[0]
            # log.debug('image_name: ', image_name)
            if image_name not in images_list:
                image_path = os.path.join(img_dir, image_name)
                if not os.path.exists(image_path):
                    log.warning('{} gets no corresponding images!'.format(result[0]))
                    continue

                images_list.append(image_name)
                ann = copy.deepcopy(ann_template)
                ann['image']['filename'] = image_name
                image = cv2.imread(image_path)
                h, w = image.shape[0], image.shape[1]
                ann['image']['size']['width'] = w
                ann['image']['size']['height'] = h

            points = [float(result[j]) for j in range(2, len(result))]
            confidence = float(result[1])
            if len(points) == 4:
                xmin, xmax, ymin, ymax = points[0], points[2], points[1], points[3]
                points = [points[0], points[1], points[2], points[1], points[2], points[3], points[0], points[3]]
            else:
                xmin = min(points[0::2])
                xmax = max(points[0::2])
                ymin = min(points[1::2])
                ymax = max(points[1::2])
            if xmax <= xmin:
                continue
            elif ymax <= ymin:
                continue
            else:
                obj= copy.deepcopy(obj_template)
                obj['name'] = label
                obj['bndbox']['xmin'] = xmin
                obj['bndbox']['xmax'] = xmax
                obj['bndbox']['ymin'] = ymin
                obj['bndbox']['ymax'] = ymax
                obj['segmentation'].append(points)
                obj['confidence'] = confidence
                ann['object'].append(obj)
            if i == len(results)-1:
                ann_dict.append(ann)
            # elif results[i+1].split(' ')[0] + '.' + ext not in images_list:
            elif results[i+1].split(' ')[0] not in images_list:
                ann_dict.append(ann)
    new_dict = filter_dict(ann_dict)
    return new_dict

def read_DOTA_v2(txt_dir, img_dir, ext='png'):
    # format: DOTA origin txt
    log.info('Start to load the txt results...')
    if not check_path(txt_dir, format='dir'):
        return None
    class_map = create_class_map_txt_result(txt_dir)
    log.info('class_map: ', class_map)
    ann_dict = []
    images_list = []
    for image_name in os.listdir(img_dir):
        image_path = os.path.join(img_dir, image_name)
        if not check_path(image_path, format='png'):
            continue
        images_list.append(image_name)
        ann = copy.deepcopy(ann_template)
        ann['image']['filename'] = image_name
        
        image = cv2.imread(image_path)
        h, w = image.shape[0], image.shape[1]
        ann['image']['size']['width'] = w
        ann['image']['size']['height'] = h
        ann_dict.append(ann)
    time.sleep(0.5)
    for filename in tqdm(os.listdir(txt_dir), desc = 'label', mininterval = mininterval, ncols = ncols):
        label = filename.replace('.txt', '')
        file_path = os.path.join(txt_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            results = f.readlines()
        # ann = copy.deepcopy(ann_template)
        for i, result in enumerate(results):
            result = result.split(' ')
            log.debug('result: ', result)
            # image_name = result[0] + '.' + ext
            image_name = result[0]
            # log.debug('image_name: ', image_name)
            if image_name not in images_list:
                if not os.path.exists(image_path):
                    log.warning('{} gets no corresponding images!'.format(result[0]))
                    continue
            idx = 0
            for i, ann in enumerate(ann_dict):
                if ann['image']['filename'] == image_name:
                    idx = i
                    break

            points = [float(result[j]) for j in range(2, len(result))]
            confidence = float(result[1])
            if len(points) == 4:
                xmin, xmax, ymin, ymax = points[0], points[2], points[1], points[3]
                points = [points[0], points[1], points[2], points[1], points[2], points[3], points[0], points[3]]
            else:
                xmin = min(points[0::2])
                xmax = max(points[0::2])
                ymin = min(points[1::2])
                ymax = max(points[1::2])
            if xmax <= xmin:
                continue
            elif ymax <= ymin:
                continue
            else:
                obj= copy.deepcopy(obj_template)
                obj['name'] = label
                obj['bndbox']['xmin'] = xmin
                obj['bndbox']['xmax'] = xmax
                obj['bndbox']['ymin'] = ymin
                obj['bndbox']['ymax'] = ymax
                obj['segmentation'].append(points)
                obj['confidence'] = confidence
                ann_dict[idx]['object'].append(obj)
    return ann_dict

def read_txt(txt_dir, img_dir, format='result', ext='jpg'):
    # format: result, annotation
    log.info('Start to load the txt dir')
    time.sleep(0.5)
    if format == 'DOTA':
        ann_dict = read_DOTA(txt_dir, img_dir, ext)
    elif format == 'result':
        ann_dict = read_txt_result(txt_dir, img_dir, ext)
    time.sleep(0.5)
    if ann_dict is not None:
        log.info('Finish loading the txt dir!')
    else:
        log.warning('Finish loading the txt dir, but get no result...')
    return ann_dict