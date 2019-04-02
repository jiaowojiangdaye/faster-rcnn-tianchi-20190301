#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 22:21:44 2019

@author: aaron
"""
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras.layers as KL
from keras.preprocessing import image as  klimage
import json
import cv2
from tools import check_path, scan_specified_files
import imgaug as ia
from imgaug import augmenters as iaa
ia.seed(1)
def task0():
    a = np.array([[[0,0,0,0,0]], [[1,1,1,1,1]], [[2,2,2,2,2]], [[3,3,3,3,3]]])
    
    #input_a = tf.placeholder(tf.int16, shape=(4,1,5), name='input_a')
    
    input_a = tf.constant(a, dtype=tf.int32)
    
    input_a = tf.squeeze(input_a)
    
    split = tf.dynamic_partition(input_a, [0,0, 1,1],2)
    [split0, split1] = split
    split11 = tf.transpose(split1)
    re= tf.matmul(split0, split11)
    
    tf.initialize_all_variables()
    sess = tf.Session()
    
    
    out_put, re = sess.run([split, re])
    
    
    
    
    print(out_put)
    
def task1():
    
    augmentation = iaa.SomeOf((0, 4), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=30),
               iaa.Affine(rotate=10),
               iaa.Affine(rotate=45)]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 3.0))
    ])
    
 
    img_path = '/media/mosay/数据/jz/cigrite/dect/data/train/useful/U型8-prat2'
    save_path = 'temp_fold'
    check_path(save_path)
    img_full_list = scan_specified_files(img_path, key='.jpg')
    for img in img_full_list:
        imgm = cv2.imread(img)
        imgs_aug = augmentation.augment_image(imgm)
        cv2.imwrite(save_path + '/'+os.path.split(img)[1], imgs_aug)



def task2():
    
    debug = True
#    
    augmentation = iaa.SomeOf((0, 5), [
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([iaa.Affine(rotate=90),
               iaa.Affine(rotate=180),
               iaa.Affine(rotate=270)]),
    iaa.Multiply((0.8, 1.5)),
    iaa.GaussianBlur(sigma=(0.0, 5.0))
])

    from  PIL import Image 
    
    image_path = '/media/mosay/数据/jz/tianchi/train/faster_rcnn/train_可视化/190109_181836_00154213.jpg'
    img = klimage.load_img(image_path)
    
    image = klimage.img_to_array(img)
    
#    plt.savefig()
    
    bboxes_resize1 = [[120, 120,180, 150],[130, 130,200, 100]]
    

    temp_aug_bbox = []
    for bbox in bboxes_resize1:
        temp_aug_bbox.append(ia.BoundingBox(x1=bbox[0], 
                                            x2=bbox[2], 
                                            y1=bbox[1], 
                                            y2=bbox[3]))
    bbs = ia.BoundingBoxesOnImage(temp_aug_bbox, shape=image.shape)
    if debug:
        pass
        plt.figure()
        plt.imshow(bbs.draw_on_image(image, thickness=2))
    
    
    seq_det = augmentation.to_deterministic()
    
    image_aug = seq_det.augment_image(image)
    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    plt.figure()
    plt.imshow(bbs_aug.draw_on_image(image_aug, thickness=2))    
    
    bboxes_resize1 = []
    for one in bbs_aug.bounding_boxes:
        bboxes_resize1.append([one.x1, one.y1, one.x2, one.y2])
    
    iimg = Image.fromarray(image_aug.astype(np.uint8))
    iimg.save('/media/mosay/数据/jz/tianchi/train/faster_rcnn/debug_doc/train_img/00.jpg')
    return bboxes_resize1



def task3():
    import tensorflow as tf
    import numpy as np
    
    rects=np.asarray([[1,2,3,4],[1,3,3,4],[1,3,4,4],[1,1,4,4],[1,1,3,4]],dtype=np.float32)
    
    scores=np.asarray([0.4,0.5,0.72,0.9,0.45],dtype=np.float32)
    
    import datetime
    with tf.Session() as sess:
    
        for i in range(50):
            old=datetime.datetime.now()
            nms = tf.image.non_max_suppression(rects,scores, max_output_size=5,iou_threshold=0.5)
            print("cost time",(datetime.datetime.now()-old).microseconds)
            # print('face detectd', len(nms.eval()))
            for index, value in enumerate(nms.eval()):
                rect = rects[value]
                print(rect)



if __name__ == '__main__':
#    task2()
    task3()

#    /media/mosay/数据/jz/tianchi/data/test/submit.json
    

    
