######################################@@@@@@@@@@@@@@#####
### TODO: for storing various types of anns
### DATE: 20210422
### AUTHOR: zzc

from collections import OrderedDict

obj_template = OrderedDict([
        ('name', 'head'),
        ('pose', None),
        ('truncated', '0'),
        ('difficult', '0'),
        ('confidence', '1'),
        ('area', '0'),
        ('bndbox', OrderedDict([
            ('xmin', '100'),
            ('ymin', '100'),
            ('xmax', '200'),
            ('ymax', '200')])),
        ('segmentation', []),
        ('mask', [])
])

ann_template = OrderedDict([
    ('object', []),
    ('image', OrderedDict([
        ('filename', './test_2.xml'),
        ('size', OrderedDict([
            ('width', '1024'),
            ('height', '1024'),
            ('depth', '3')]))
    ]))
])

voc_template = OrderedDict([
    ('annotation', OrderedDict([
        ('folder', 'train'),
        ('filename', './test_2.xml'),
        ('source', OrderedDict([
            ('database', 'the plane part database'),
            ('annotation', 'PASCAL VOC2007'),
            ('image', 'i do not know'),
            ('flickrid', '123')])),
        ('owner', OrderedDict([
            ('department', 'AIR'),
            ('name', 'zzc')])),
        ('size', OrderedDict([
            ('width', '1024'),
            ('height', '1024'),
            ('depth', '3')])),
        ('segmented', '0'),
        ('object', [])
    ]))
])

HRSC_template = OrderedDict([
    ('HRSC_Image', OrderedDict([
        ('Img_ID', '1'),
        ('Img_FileName', 'test'),
        ('Img_FileFmt', 'jpg'),
        ('Img_SizeWidth', '1024'),
        ('Img_SizeHeight', '1024'),
        ('Img_SizeDepth', '3'),
        ('HRSC_Objects', OrderedDict([
            ('HRSC_Object', [])
        ]))
    ]))
])