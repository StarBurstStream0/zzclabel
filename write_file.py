######################################@@@@@@@@@@@@@@#####
### TODO: for saving all parsed anns
### DATE: 20210423
### AUTHOR: zzc

import os
import xmltodict
import json as js
import copy
import codecs
import numpy as np
from template import *
from tqdm import tqdm
import time
import shutil
from zzclog import *
log = ZZCLOG(2)

# tqdm parameters
desc = 'Transfered annotations'
mininterval = 0.2
ncols = 100

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
    elif format == 'dir':
        pass
    return 1

def write_voc(ann_dict, save_path):
    log.info('Start to save parsed labels into voc format...')
    time.sleep(0.5)
    for ann in tqdm(ann_dict, desc = desc, mininterval = mininterval, ncols = ncols):
        xml_name = ann['image']['filename'][:-3]+'xml'
        xml_path = os.path.join(save_path, xml_name)
        # logdebug(xml_path)
        xml_dict = copy.deepcopy(voc_template)
        xml_dict['annotation']['folder'] = save_path
        xml_dict['annotation']['filename'] = ann['image']['filename']
        xml_dict['annotation']['size']['width'] = ann['image']['size']['width']
        xml_dict['annotation']['size']['height'] = ann['image']['size']['height']
        xml_dict['annotation']['object'] = ann['object']

        xml = xmltodict.unparse(xml_dict, encoding='utf-8', pretty=True)
        with codecs.open(xml_path, 'w', 'utf-8') as f:
            f.write(xml)
            # logdebug('Finish loading {}!'.format(xml_name))
        # break
    time.sleep(0.5)
    log.info('Finish saving all {} instances!'.format(len(ann_dict)))

def eight2five(bbox):
    """
    :param bbox: The polygon stored in format [x1, y1, x2, y2, x3, y3, x4, y4]
            shape [num_boxes, 8]
    :return: Rotated Rectangle in format [cx, cy, w, h, theta]
            shape [num_rot_recs, 5]
    """
    # print('bbox before: ', bbox.shape)
    bbox = np.array(bbox,dtype=np.float32)
    bbox = np.reshape(bbox,newshape=(2, 4),order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # print('bbox after: ', bbox.shape)
    angle = np.arctan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    # angle = np.arctan2(-(bbox[:, 0,1]-bbox[:, 0,0]),bbox[:, 1,1]-bbox[:, 1,0])
    # center = [[0],[0]] ## shape [2, 1]
    # print('angle: ', angle)
    center = np.zeros((2, 1))
    for i in range(4):
        center[0, 0] += bbox[0,i]
        center[1, 0] += bbox[1,i]

    center = np.array(center,dtype=np.float32)/4.0

    # R = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], dtype=np.float32)

    normalized = np.matmul(R.transpose((1, 0)),bbox-center)


    xmin = np.min(normalized[0, :], axis=0)
    # print('diff: ', (xmin - normalized[:, 0, 3]))
    # assert sum((abs(xmin - normalized[:, 0, 3])) > eps) == 0
    xmax = np.max(normalized[0, :], axis=0)
    # assert sum(abs(xmax - normalized[:, 0, 1]) > eps) == 0
    # print('diff2: ', xmax - normalized[:, 0, 1])
    ymin = np.min(normalized[1, :], axis=0)
    # assert sum(abs(ymin - normalized[:, 1, 3]) > eps) == 0
    # print('diff3: ', ymin - normalized[:, 1, 3])
    ymax = np.max(normalized[1, :], axis=0)
    # assert sum(abs(ymax - normalized[:, 1, 1]) > eps) == 0
    # print('diff4: ', ymax - normalized[:, 1, 1])

    w = xmax - xmin + 1
    h = ymax - ymin + 1
    # TODO: check it

    angle = angle % ( 2 * np.pi)

    # dboxes = np.concatenate((center[:, 0].astype(np.float), center[:, 1].astype(np.float), w, h, angle), axis=1)
    dboxes = [float(center[0]), float(center[1]), float(w), float(h), float(angle)]
    return dboxes

def write_HRSC(ann_dict, save_dir):
    log.info('Start to save parsed labels into HRSC format...')
    time.sleep(0.5)
    images_list = []
    # xml_dict = copy.deepcopy(HRSC_template)
    for ann in tqdm(ann_dict, desc = desc, mininterval = mininterval, ncols = ncols):
        filename = ann['image']['filename']
        xml_name = filename[:-3]+'xml'
        xml_path = os.path.join(save_dir, xml_name)
        log.debug(xml_path)
        # if filename not in images_list:
        #     images_list.append(filename)
        xml_dict = copy.deepcopy(HRSC_template)
        xml_dict['HRSC_Image']['Img_FileName'] = filename[:-4]
        xml_dict['HRSC_Image']['Img_FileFmt'] = filename[-3:]
        xml_dict['HRSC_Image']['Img_SizeWidth'] = ann['image']['size']['width']
        xml_dict['HRSC_Image']['Img_SizeWidth'] = ann['image']['size']['height']
        xml_dict['HRSC_Image']['Img_SizeDepth'] = ann['image']['size']['depth']
        for obj in ann['object']:
            HRSC_obj = {}
            HRSC_obj['Class_ID'] = obj['name']
            HRSC_obj['box_xmin'] = obj['bndbox']['xmin']
            HRSC_obj['box_ymin'] = obj['bndbox']['ymin']
            HRSC_obj['box_xmax'] = obj['bndbox']['xmax']
            HRSC_obj['box_ymax'] = obj['bndbox']['ymax']
            obb = obj['segmentation'][0]
            rbb = eight2five(obb)
            HRSC_obj['mbox_cx'] = rbb[0]
            HRSC_obj['mbox_cy'] = rbb[1]
            HRSC_obj['mbox_w'] = rbb[2]
            HRSC_obj['mbox_h'] = rbb[3]
            HRSC_obj['mbox_ang'] = rbb[4]
            xml_dict['HRSC_Image']['HRSC_Objects']['HRSC_Object'].append(HRSC_obj)
            # logdebug(xml_path)

        xml = xmltodict.unparse(xml_dict, encoding='utf-8', pretty=True)
        with codecs.open(xml_path, 'w', 'utf-8') as f:
            f.write(xml)
            # logdebug('Finish loading {}!'.format(xml_name))
        # break
    time.sleep(0.5)
    with codecs.open(os.path.join(save_dir, 'images.txt'), 'w', 'utf-8') as f:
        for image in images_list:
            f.write(image.split('.')[0]+'\n')
        log.info('Finish saving images txt!')
    log.info('Finish saving all {} images!'.format(len(ann_dict)))

def write_xml(ann_dict, save_dir, format='voc'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if format == 'voc':
        return write_voc(ann_dict, save_dir)
    elif format == 'HRSC':
        return write_HRSC(ann_dict, save_dir)

def filter_class(ann_dict):
    classes = []
    for ann in ann_dict:
        for obj in ann['object']:
            classes.append(obj['name'])
    return list(set(classes))

def write_json(ann_dict, save_path, format='COCO'):
    log.info('Start to save parsed labels into json format...')
    time.sleep(0.5)
    data_dict = {}
    data_dict['images'] = []
    images_name = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    cls_names = filter_class(ann_dict)
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 0
    image_id = 0
    with open(save_path, 'w') as f_out:
        for ann in tqdm(ann_dict, desc = desc, mininterval = mininterval, ncols = ncols):
            # log.debug('ann[image][filename]: ', ann['image']['filename'])
            # log.debug('ann[object]: ', ann['object'])
            single_image = {}
            single_image['file_name'] = ann['image']['filename']
            single_image['id'] = image_id
            single_image['width'] = ann['image']['size']['width']
            single_image['height'] = ann['image']['size']['height']
            data_dict['images'].append(single_image)

            for obj in ann['object']:
                # if ann['image']['filename'] == 'P0161__1__0___0__1__0___0.png':
                # log.debug('ann[object]: ', ann['object'])
                log.debug('ann[image][filename]: ', ann['image']['filename'])
                log.debug('obj: ', obj)
                # annotations
                single_obj = {}
                name = obj['name']
                single_obj['category_id'] = cls_names.index(name) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['segmentation'][0])
                xmin, xmax, ymin, ymax = int(obj['bndbox']['xmin']), int(obj['bndbox']['xmax']), \
                    int(obj['bndbox']['ymin']), int(obj['bndbox']['ymax'])
                single_obj['area'] = (xmax - xmin) * (ymax - ymin)
                single_obj['iscrowd'] = 0
                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        js.dump(data_dict, f_out)
    time.sleep(0.5)
    log.info('Finish saving all {} instances! {} images processed!'.format(inst_count, image_id))

def create_class_map(ann_dict):
    categories = []
    for ann in ann_dict:
        for obj in ann['object']:
            categories.append(obj['name'])
    categories = list(set(categories))
    class_map = {}
    for i, category in enumerate(categories):
        class_map[category] = i
    return class_map

def write_darknet(ann_dict, image_dir, save_dir, ext='jpg'):
    log.info('Start to save parsed labels into darknet format...')
    class_map = create_class_map(ann_dict)
    img_list = []

    time.sleep(0.5)
    for ann in tqdm(ann_dict, desc = desc, mininterval = mininterval, ncols = ncols):
        txt_name = ann['image']['filename'][:-3]+'txt'
        img_name = ann['image']['filename'][:-3]+ext
        txt_path = os.path.join(save_dir, txt_name)
        img_path = os.path.join(save_dir, img_name)
        img_list.append(img_name)

        w = ann['image']['size']['width']
        h = ann['image']['size']['height']

        with codecs.open(txt_path, 'w', 'utf-8') as f:
            for obj in ann['object']:
                label_id = class_map[obj['name']]
                xmax, xmin, ymax, ymin = obj['bndbox']['xmax'], obj['bndbox']['xmin'], obj['bndbox']['ymax'], obj['bndbox']['ymin']
                cx_ratio = (xmax + xmin) / (2*w)
                cy_ratio = (ymax + ymin) / (2*h)
                w_ratio = (xmax - xmin) / w
                h_ratio = (ymax - ymin) / h
                f.write('%d %.6f %.6f %.6f %.6f\n' % (label_id, cx_ratio, cy_ratio, w_ratio, h_ratio))

    time.sleep(0.5)
    log.info('Finish saving all {} instances!'.format(len(ann_dict)))
    with codecs.open(os.path.join(save_dir, 'data.txt'), 'w', 'utf-8') as f:
        for img_name in img_list:
            f.write(os.path.join(save_dir, img_name)+'\n')
            shutil.copyfile(os.path.join(image_dir, img_name), os.path.join(save_dir, img_name))
    log.info('Finish saving data.txt file!')
    with codecs.open(os.path.join(save_dir, 'data.names'), 'w', 'utf-8') as f:
        for category in class_map:
            f.write(category+'\n')
    log.info('Finish saving data.names file!')        

def write_txt(ann_dict, save_dir, image_dir=None, format='darknet', ext='jpg'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if format == 'darknet':
        return write_darknet(ann_dict, image_dir, save_dir, ext)
    elif format == 'DOTA':
        return None