######################################@@@@@@@@@@@@@@#####
### TODO: for cropping images and labels
### DATE: 20210426
### AUTHOR: zzc

import os
import codecs
import numpy as np
import math
import cv2
import copy
from multiprocessing import Pool
from functools import partial
import shapely.geometry as shgeo
from template import *
from tqdm import tqdm
import time
from copy import deepcopy
from zzclog import *
log = ZZCLOG(2)

# tqdm parameters
desc = 'Loaded images'
mininterval = 0.2
ncols = 100

def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))



class crop_file():
    def __init__(self,
                 ann_dict,
                 imagepath,
                 outimagepath,
                 code = 'utf-8',
                 gap=512,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=False,
                 ext = 'png',
                 padding=True,
                 num_process=8
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        :param padding: if to padding the images so that all the images have the same size
        """

        self.ann_dict = ann_dict
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = imagepath

        self.outimagepath = outimagepath
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding
        self.pool = Pool(num_process)
        log.warning('padding:', padding)

        self.cropped_ann_dict = []

        if not os.path.isdir(self.outimagepath):
            os.mkdir(self.outimagepath)

    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + '.' + self.ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
                outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
                count = count + 1
            elif (count == (pos + 1)%5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def savepatches_v2(self, resizeimg, objects, subimgname, left, up, right, down):
        # log.debug('Runing into savepatches v2...')
        # log.debug('subimgname: ', subimgname)
        # log.debug('objects: ', objects)
        # log.debug('left: ', left)
        # log.debug('up: ', up)
        
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])

        ann = copy.deepcopy(ann_template)

        ann['image']['filename'] = subimgname + '.' + self.ext
        ann['image']['size']['width'] = self.subsize
        ann['image']['size']['height'] = self.subsize

        for obj in objects:
            obj_crop = copy.deepcopy(obj)
            seg = obj_crop['segmentation'][0]
            # seg = list(seg)
            gtpoly = shgeo.Polygon([(int(seg[i]), int(seg[i+1])) \
                for i in range(0, len(seg), 2)])
            if (gtpoly.area <= 0):
                log.debug('Deleted obj[name]: ', obj['name'])
                continue
            # log.debug('gtpoly: ', gtpoly)
            inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

            if (half_iou == 1):
                seg_ = []
                for i in range(len(seg)):
                    if i % 2 == 0:
                        seg_.append(seg[i] - left)
                    else:
                        seg_.append(seg[i] - up)
                obj_crop['segmentation'][0] = seg_
                #############
                obj_crop['bndbox']['xmax'] = max(seg_[0::2])
                obj_crop['bndbox']['xmin'] = min(seg_[0::2])
                obj_crop['bndbox']['ymax'] = max(seg_[1::2])
                obj_crop['bndbox']['ymin'] = min(seg_[1::2])
                obj_crop['area'] = (max(seg_[1::2]) - min(seg_[1::2])) * (max(seg_[0::2]) - min(seg_[0::2]))
                #############
                # log.debug('obj: ', obj)
                ann['object'].append(obj_crop)

            elif (half_iou > 0):
                if (half_iou > self.thresh):
                    seg_ = []
                    for i in range(len(seg)):
                        if i % 2 == 0:
                            seg_.append(seg[i] - left)
                        else:
                            seg_.append(seg[i] - up)
                    obj_crop['segmentation'][0] = seg_
                    #############
                    obj_crop['bndbox']['xmax'] = max(seg_[0::2])
                    obj_crop['bndbox']['xmin'] = min(seg_[0::2])
                    obj_crop['bndbox']['ymax'] = max(seg_[1::2])
                    obj_crop['bndbox']['ymin'] = min(seg_[1::2])
                    obj_crop['area'] = (max(seg_[1::2]) - min(seg_[1::2])) * (max(seg_[0::2]) - min(seg_[0::2]))
                    #############
                    # log.debug('obj: ', obj)
                    ann['object'].append(obj_crop)

        # log.debug('ann: ', ann)
        self.cropped_ann_dict.append(ann)
        # log.debug('cropped_ann_dict: ', self.cropped_ann_dict)

        self.saveimagepatches(resizeimg, subimgname, left, up)

    def crop_single(self, base_name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        image_name = base_name + '.' + extent
        try:
            img = cv2.imread(os.path.join(self.imagepath, image_name))
            # img = cv2.imread(os.path.join(self.imagepath, name))
            # log.info('img name:', image_name)
        except:
            log.debug('img name:', image_name)
        if np.shape(img) == ():
            log.error('No image loaded!')
            return
        # fullname = os.path.join(self.labelpath, name + '.txt')
        # objects = util.parse_dota_poly2(fullname)
        for ann in self.ann_dict:
            if ann['image']['filename'] == image_name:
                objects = [deepcopy(x) for x in ann['object']]
        # log.debug('image_name; ', image_name)
        # log.debug('objects; ', objects)
        for obj in objects:
            # log.debug('obj[segmentation]: ', obj['segmentation'])
            obj['segmentation'][0] = [float(x)*rate for x in obj['segmentation'][0]]
            # obj['segmentation'] = list(map(lambda x:rate*x, obj['segmentation']))
            # log.debug('obj[segmentation]: ', obj['segmentation'])
            # obj['segmentation'][0] = [int(x) for x in obj['segmentation'][0]]

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = base_name + '__' + str(rate) + '__'
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        # if (max(weight, height) < self.subsize):
        #     return

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '___' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                log.debug('subimgname: ', subimgname)
                self.savepatches_v2(resizeimg, objects, subimgname, left, up, right, down)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def crop_data(self, rate):
        """
        :param rate: resize rate before cut
        """

        log.info('Start to crop images and labels...')
        time.sleep(0.5)
        self.cropped_ann_dict = []
        imagenames = [ann['image']['filename'].split('.')[0] for ann in self.ann_dict]

        # worker = partial(self.crop_single, rate=rate, extent=self.ext)
        #
        for name in tqdm(imagenames, desc = desc, mininterval = mininterval, ncols = ncols):
            log.debug('name: ', name)
            self.crop_single(name, rate, self.ext)
        # self.pool.map(worker, imagenames)
        time.sleep(0.2)
        log.info('Finish cropping the images and labels!')
        # log.debug('self.cropped_ann_dict: ', self.cropped_ann_dict)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)