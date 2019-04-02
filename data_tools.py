#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:24:58 2019

@author: mosay
"""

from tools import scan_specified_files
#from matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import time
import json
import shutil
from PIL import Image,ImageDraw,ImageFont
import sys
from tools import check_path, scan_specified_files, compare_two_fold
from keras.preprocessing import image
from ThisConfig import ThisConfig



def drew_detect_resualt(path, save_path, bboxes, category_ids, class_info, debug=False):
    
    img = Image.open(path)
    if len(category_ids)==0:
        print("image: no bbox", path)
    ttfont = ImageFont.truetype('lib/华文细黑.ttf', 10)
    for idx, bbox in enumerate(bboxes):
        
        category_id = category_ids[idx]
        class_name = class_info[category_id]['name']
        draw = ImageDraw.Draw(img)
        draw.line([(bbox[0],bbox[1]),
               (bbox[2],bbox[1]),
               (bbox[2],bbox[3]),
               (bbox[0],bbox[3]),
               (bbox[0],bbox[1])], width=3, fill='yellow')
#            print('class_nameclass_nameclass_nameclass_name',class_name)
#            unicode('杨','utf-8')
        draw.text((bbox[0],bbox[1]), class_name, 
                  fill=(255,0,0), font= ttfont)
#    if debug:
#        img.show()
        
    img.save(save_path)


def get_img_mean():
    img_root  = '/media/mosay/数据/jz/tianchi/data/train/jinnan2_round1_train_20190222'
    
    img_list = scan_specified_files(img_root, key='.jpg')
    
    pixel_sum = np.zeros([3], np.float)
    for idx, img_full_path in enumerate(img_list):
        img = Image.open(img_full_path)
        w, h = img.size
        sum1 = np.sum(img, axis=0)
        sum1 = np.sum(sum1, axis=0)
        sum1 = sum1/(w*h)
        pixel_sum += sum1
        if idx%200:
            print('processing ', idx)
    avg_mean = pixel_sum/len(img_list)
    
    print(avg_mean)


def read_index(file_path):
    with open(file_path, 'r') as f:
        index = f.readlines()
        index_ = [int(x) for x in index]
        return index_

def parse_data(annotations):
    '''
    annotations_ = {1: {'bboxes':[], 'category_ids':[]},
                    2: {'bboxes':[], 'category_ids':[]},
                    3: {'bboxes':[], 'category_ids':[]}}
    '''
    annotations_ = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        if image_id not in annotations_:
            annotations_[image_id] = {'bboxes':[], 'category_ids':[]}
            
        bbox = annotation['bbox']
        bbox[2] = bbox[0]+bbox[2]
        bbox[3] = bbox[1]+bbox[3]
        annotations_[image_id]['bboxes'].append(bbox)
        annotations_[image_id]['category_ids'].append(annotation['category_id'])
    
    return annotations_

def prepare_class( categories):
    
    class_map = {}
    class_num = 5+1
    for i in range(class_num-1): #TODO
        class_map[categories[i]['id']] = categories[i]['name']
    return class_map

def split_train_data():
    config = ThisConfig()
    
    debug = False
    
    
    split_record = '/media/mosay/数据/jz/tianchi/train/faster_rcnn/split_record_old'
    
    postive_train_index = []
    postive_val_index = []
    nagetive_train_index = []
    nagetive_val_index = []
    postive_index = read_index(os.path.join(split_record, 'postive_index.txt'))
    nagetive_index = read_index(os.path.join(split_record, 'nagetive_index.txt'))
    
    split_idx0 = int(len(postive_index)*(1.0 - config.VAL_DATA_RATE))
    split_idx1 = int(len(nagetive_index)*(1.0 - config.VAL_DATA_RATE))
    
    postive_train_index = postive_index
    postive_val_index = postive_index[split_idx0: ]
    nagetive_train_index = []
    nagetive_val_index = nagetive_index[split_idx1: ]
    
    f = open(config.annotion_path, encoding='utf-8')
    dataset = json.load(f)
    annotations = dataset['annotations']
    annotations = parse_data(annotations)
    image_infos = dataset['images']
    prepare_class(dataset['categories'])
    
    
    save_train_fold='/media/mosay/数据/jz/tianchi/data/train/train_old'
    
    
    restricted_img_dir = config.restricted_img_dir
    
    postive_index = postive_train_index
    nagetive_index = nagetive_train_index
    
    
    
    save_restricted_img_dir = save_train_fold+'/restricted'
    check_path(save_restricted_img_dir)
    error_img_path = '/media/mosay/数据/jz/tianchi/data/train/error'
    f = open('dataset_log.txt', 'w')
    time0 = time.time()
    counter = 0
    for idx in postive_index:
        new_json = {}
        
        info = image_infos[idx]
        image_id=info['id']
        width=info['width']
        height=info['height']
        img_full_path = os.path.join(restricted_img_dir, info['file_name'])
        if idx not in annotations:                                          #TODO
            print('this img no bboxes:', info['file_name'])
            shutil.copy(img_full_path, os.path.join(error_img_path, info['file_name']))
            continue
        else:
            elsement = annotations[idx]
            category_ids = elsement['category_ids']
            bboxes = elsement['bboxes']

        if 'need_check_per_image' == 'need_check_per_image':
            try:
                img = image.load_img(img_full_path)
            except FileNotFoundError as e:
                print(e)
#                    print(annotation.image_name)
                f.writelines(str(idx) + ' : ' + img_full_path + '\n')
                continue
            width_gt, height_gt = img.size  #TODO
            if [width, height] != [width_gt, height_gt]:
                print('wrong width and height')
                f.writelines(str(image_id) + ': wrong width and height: ',img_full_path)
                sys.exit()
                continue
        
        objects = []
        
        for idx_, bbox in enumerate(bboxes): 
            oneobject = {}
            
            x1 = min(bbox[0], bbox[2])
            y1 = min(bbox[1], bbox[3])
            x2 = max(bbox[0], bbox[2])
            y2 = max(bbox[1], bbox[3])
            if x1 >= x2 or y1 >= y2:
                print('bbox_gt error ',bbox )
                continue
            
            oneobject['label'] = category_ids[idx_]
            oneobject['bbox'] = bbox
            
            objects.append(oneobject)
        
        
        
        shutil.copy(img_full_path, save_restricted_img_dir+'/'+info['file_name'])
        new_json['width'] = width
        new_json['height'] = height
        new_json['depth'] = 3
        new_json['file_name'] = info['file_name']
        new_json['objects'] = objects
        
        new_json_path = os.path.join(save_restricted_img_dir, info['file_name'].split('.')[0]+'.json')
        
        new_json_f = open(new_json_path,'w',encoding='utf-8')
        json.dump(new_json, new_json_f)
        new_json_f.close()
        
        re_category_ids = []
        re_bboxes = []
        for idx_, bbox in enumerate(bboxes):
            x1 = min(bbox[0], bbox[2])
            y1 = min(bbox[1], bbox[3])
            x2 = max(bbox[0], bbox[2])
            y2 = max(bbox[1], bbox[3])
            if x1 >= x2 or y1 >= y2:
                print('bbox_gt error ',bbox )
                continue
            re_category_ids.append(category_ids[idx_])
            rect = []
            rect.append(x1)
            rect.append(y1)
            rect.append(x2)
            rect.append(y2)
            re_bboxes.append(rect)
            
            
        if debug:
            
            save_path = 'train_data_virsual_fold'
            check_path(save_path)
            drew_detect_resualt(img_full_path, 
                                os.path.join(save_path, img_full_path.split('/')[-1]), 
                                re_bboxes, 
                                re_category_ids, 
                                class_map, 
                                debug)
        

        
        counter += 1
        step=200
        if counter % step == 0:
            rest_time = (time.time()-time0)*((len(annotations)-counter)/(step))
            print('----Adding the image:', counter, 
                  'rest time(sec) = ', rest_time)
            time0 = time.time()

    
    #%% nagetive samples
    save_normal_img_dir = save_train_fold+'/normal'
    normal_img_dir = config.normal_img_dir
    img_name_list = sorted(os.listdir(normal_img_dir))
    
    for idx in nagetive_index:
        new_json = {}
        img_name = img_name_list[idx]
        img_full_path = os.path.join(normal_img_dir, img_name)
        if 'need_check_per_image' == 'need_check_per_image':
            try:
                img = image.load_img(img_full_path)
            except FileNotFoundError as e:
                print(e)
#                    print(annotation.image_name)
                f.writelines(str(idx) + ' : ' + img_name + '\n')
                continue
        width, height = img.size  #TODO

        
        

        shutil.copy(img_full_path, save_normal_img_dir+'/'+img_name)
        new_json['width'] = width
        new_json['height'] = height
        new_json['depth'] = 3
        new_json['objects'] = []
        new_json['file_name'] = img_name
        
        new_json_path = os.path.join(save_normal_img_dir, img_name.split('.')[0]+'.json')
        
        new_json_f = open(new_json_path,'w',encoding='utf-8')
        json.dump(new_json, new_json_f)
        new_json_f.close()


        counter += 1
        step=500
        if counter % step == 0:
            rest_time = (time.time()-time0)*((len(img_name_list)-counter)/(step))
            print('----Adding :', counter, 
                  'rest time(sec) = ', rest_time)
            time0 = time.time()

        
    f.close()
    print('-----------loaded total image ----------------:', counter)

def compare_data():
    import shutil, random
    path1 = '/media/mosay/数据/jz/tianchi/data/train/jinnan2_round1_train_20190222/restricted'
    path2 = '/media/mosay/数据/jz/tianchi/data/train/jinnan2_round1_train_20190305/restricted'
    info1, info2 = compare_two_fold(path1,  path2)
    
    index = range(len(info1['over']))
    random.shuffle(index)
    
    for i in index[:130]:
        img_name = info1['over'][i]
        shutil.move(os.path.join(path1, img_name), os.path.join(save_path, img_name))
    
    return
    
if __name__ == '__main__':
    split_train_data()

#    compare_data()
    
    