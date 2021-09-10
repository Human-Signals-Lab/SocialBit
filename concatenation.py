#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 19:23:33 2021

@author: dawei
"""
"""
Load audio feature/voc embedding vectors (1 sec), do concatenation for every k sec and save the segments
"""

import pandas as pd
import os
import numpy as np
import csv
from collections import Counter

import check_dirs

np.random.seed(0)

#%%

def load_csv(csv_list):
    """
    Obtain feat and labels from list of csv spectrogram
    
    args:
        list of csv paths
    return:
        data and labels and time stamps in lists  
    """
    
    feat, labels, timestamp = [], [], []
    for csv_item in csv_list:
        feat_temp = pd.read_csv(csv_item, header=None).values
        # naming format of the csv: /../seg<x>_<label>.csv
        label = csv_item.split('/')[-1].split('_')[1].replace('.csv', '')
        time = int(csv_item.split('/')[-1].split('_')[0].replace('seg', '')) + 1   # from 0-index
        #print('loaded feature shape:', spectrogram.shape, 'label:', label, 'timestamp:', time)
        feat.append(feat_temp)
        labels.append(label)
        timestamp.append(time)
    return feat, labels, timestamp


def label_convert(labels):  
    """
    Change sound labels to a consistent format, '1' for wearer, '2' for non-wearer, '0' for back, 'x' for ambiguous 
    """
    
    idx_wearer_voice = np.where(labels == '1')   # wearer
    labels[idx_wearer_voice] = '1'
    idx_other_voice = np.where(labels == 'm')   # mixed counted as wearer
    labels[idx_other_voice] = '1'
    
    idx_other_voice = np.where(labels == '2')   # non-wearer human
    labels[idx_other_voice] = '2'
    idx_other_voice = np.where(labels == 'c')   # # non-wearer baby crying
    labels[idx_other_voice] = '2'
    
    idx_other_voice = np.where(labels == 'p')
    labels[idx_other_voice] = '0'   
    idx_other_voice = np.where(labels == 't')
    labels[idx_other_voice] = '0' 
    idx_noise = np.where(labels == 'b')   # non-vocal background
    labels[idx_noise] = '0'
    
    idx_amb = np.where(labels == 'x')   # non-vocal background, randomize as '1' or '2'
    for i in idx_amb[0]:
        labels[i] = str(np.random.randint(1,3))
    
    return labels

def mode(test_list):
    """
    return list of modes for an array
    """
    # Multimode of List
    # using loop + formula 
    res = []
    test_list1 = Counter(test_list) 
    temp = test_list1.most_common(1)[0][1] 
    for ele in test_list:
      if test_list.count(ele) == temp:
        res.append(ele)
    res = list(set(res))
    return res

#%%    
'''load features and labels for every second'''

target_file_list = ['call', 'reading', 'game', 'TV', 'outdoor', 'dinner']   
sub_list = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11']

seg_size = 30   # in sec
seg_hop = 5
embedding = True   # whether to cat emb or acoustic feat


for sub in sub_list:
    for ac in target_file_list:
        if embedding:
            feat_dir = '/media/hd4t1/dawei/socialbit/field_study/field_data/%s/embedding_1sec/%s/' % (sub, ac)   # dir to load 1-sec embeddings
            save_seg_dir = '/media/hd4t1/dawei/socialbit/field_study/field_data/%s/seg_30sec_emb/%s/' % (sub, ac)   # dir to save segments
            
        else:
            feat_dir = '/media/hd4t1/dawei/socialbit/field_study/field_data/%s/feat_1sec/%s/' % (sub, ac)   # dir to load 1-sec acoustic feat
            save_seg_dir = '/media/hd4t1/dawei/socialbit/field_study/field_data/%s/seg_30sec_feat/%s/' % (sub, ac)   # dir to save segments       
        
        csv_list = [os.path.join(feat_dir, item) \
                    for item in os.listdir(feat_dir) if item.endswith('.csv')]
        # sort the feature instances by second, naming format of the csv: /../seg<x>_<label>.csv
        csv_list = sorted(csv_list, key=lambda x:int(x.split('/')[-1].split('_')[0].replace('seg', '')))
        
        feat, labels, timestamp = load_csv(csv_list)
        feat = np.asarray(feat)
        # inpt: [#, 1, 1000], out: [#, 1000, 1]
        if embedding:
            feat_temp = []
            for i in range(len(feat)):
                feat_temp.append(feat[i].T)
            feat_temp = np.asarray(feat_temp)
            feat = feat_temp
            del feat_temp

        labels = label_convert(np.asarray(labels))
        
        assert len(labels) == len(feat), 'size of labels and feat not alligned'
        assert len(labels) == len(timestamp), 'size of labels and timestamps not alligned'
        print('\nloaded sub, interaction type:', sub, ac)
        
        
        '''concatenation and save'''
       
        for t in range(0, len(labels)-seg_hop, seg_hop):  
            # ignore the last seg if less than the defined seg size
            if len(feat[t:,:,:]) < seg_size:
                break
            
            if not embedding:
                seg_feat = np.empty((48, 0))   # segment features, out: [48, 20*seg_size], where there are 20 frames per sec
            else:
                seg_feat = np.empty((1000, 0))   # for emb out: [1000, seg_size]
            # stack feat/emd for every second
            for i in range(seg_size):
                seg_feat = np.hstack((seg_feat, feat[t+i,:,:]))  
            frame_labels = np.copy(labels[t:t+seg_size]).astype(int)   # frame labels in a segment
            time = np.copy(timestamp[t])
            assert time == t+1, 'time mismatch'
            
            # get the cnt of labels in a seg
            counter = Counter(frame_labels)
            cnt_0, cnt_1, cnt_2 = counter[0], counter[1], counter[2]
            # social isolation seg
#            if cnt_1 <= 5:
#                seg_label = 0
#            # mono speech seg
#            elif cnt_2 <= 5:
#                seg_label = 1
#            # conversation seg
#            else:
#                seg_label = 2
                
#            # social isolation seg
#            if cnt_1 <= 2:
#                seg_label = 0
#            # mono speech seg
#            elif cnt_1 >= 8:
#                seg_label = 1
#            # conversation or mono speech seg
#            else:
#                # 1st mode of frame_labels
#                label_mode = mode(list(frame_labels))[0]
#                if label_mode == 1:
#                    seg_label = 1
#                elif label_mode == 2:
#                    seg_label = 2
#                else:
#                    seg_label = 1
            
            if cnt_1 < 5:
                seg_label = 0
            else:
                if ac in ['call', 'reading']:
                    seg_label = 1
                else:
                    seg_label = 2
            
#            if cnt_1 < 5:
#                seg_label = 0
#            else:
#                if ac in ['call', 'reading']:
#                    seg_label = 1
#                elif ac == 'game':
#                    seg_label = 2
#                elif ac in ['TV', 'dinner']:
#                    seg_label = 3
#                else:
#                    seg_label = 4
                
            #print(seg, seg_label)
            
            check_dirs.check_dir(save_seg_dir)
            with open(save_seg_dir + str(t+seg_size) + 'sec_' + str(seg_label) + ".csv", 'w', newline='') as csvfile:   # no gap in lines
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerows(seg_feat)
            csvfile.close()                  
        # check the last one   
        print('Done. Check segment feature shape:', seg_feat.shape, 'subject:', sub, 'class', ac)
