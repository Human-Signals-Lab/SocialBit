#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 00:46:03 2021

@author: dawei
"""

import pandas as pd
import numpy as np
import os
from collections import Counter

from sklearn.model_selection import StratifiedKFold, LeavePGroupsOut, GroupKFold
from numpy.random import seed
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.random import set_seed
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

import check_dirs

seed(0)
# keras seed
set_seed(0)

#%%
class model_2class():
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def model(self):    
        model = Sequential()
        model.add(L.Conv1D(32, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(L.Conv1D(64, strides=2, kernel_size=4, activation='relu', padding='same'))
        model.add(L.Flatten())
        model.add(L.Dense(64, activation='relu'))
        model.add(L.Dense(64, activation='relu'))
        model.add(L.Dense(1, activation='sigmoid'))
    
        # Compile the model
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=binary_crossentropy,
                      optimizer=opt,
                      metrics=['binary_accuracy'])
        return model

#class model_2class():
#    def __init__(self, input_shape):
#        self.input_shape = input_shape
#    def model(self):    
#        model = Sequential()
#        model.add(L.Conv1D(64, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
#        model.add(L.Conv1D(128, strides=2, kernel_size=4, activation='relu', padding='same'))
#        model.add(L.Conv1D(256, strides=2, kernel_size=4, activation='relu', padding='same'))
#        model.add(L.Conv1D(512, strides=2, kernel_size=4, activation='relu', padding='same'))
#        model.add(L.Flatten())
#        model.add(L.Dense(1024, activation='relu'))
#        model.add(L.Dropout(0.2))
#        model.add(L.Dense(128, activation='relu'))
#        model.add(L.Dropout(0.2))
#        model.add(L.Dense(1, activation='sigmoid'))
#    
#        # Compile the model
#        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#        model.compile(loss=binary_crossentropy,
#                      optimizer=opt,
#                      metrics=['binary_accuracy'])
#        return model
    
class model_2class2D():
    def __init__(self, input_shape):
        self.input_shape = input_shape
    def model(self):    
        model = Sequential()
        model.add(L.Conv2D(64, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(L.Conv2D(128, strides=2, kernel_size=4, activation='relu', padding='same'))
        model.add(L.Conv2D(256, strides=2, kernel_size=4, activation='relu', padding='same'))
        model.add(L.Conv2D(512, strides=2, kernel_size=4, activation='relu', padding='same'))
        model.add(L.Flatten())
        model.add(L.Dense(1024, activation='relu'))
        model.add(L.Dropout(0.2))
        model.add(L.Dense(128, activation='relu'))
        model.add(L.Dropout(0.2))
        model.add(L.Dense(1, activation='sigmoid'))
    
        # Compile the model
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=binary_crossentropy,
                      optimizer=opt,
                      metrics=['binary_accuracy'])
        return model
    
class model_3class():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    def model(self):    
        model = Sequential()
        model.add(L.Conv2D(32, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
        #model.add(L.MaxPooling1D(pool_size=(2, 2)))
        model.add(L.Conv2D(64, strides=2, kernel_size=4, activation='relu', padding='same'))
        #model.add(L.MaxPooling1D(pool_size=(2, 2)))
        model.add(L.Flatten())
        model.add(L.Dense(64, activation='relu'))
        model.add(L.Dense(64, activation='relu'))
        model.add(L.Dense(num_classes, activation='softmax'))
    
        # Compile the model
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])
        return model
      

#class model_3class():
#    def __init__(self, input_shape):
#        self.input_shape = input_shape
#    def model(self):    
#        model = Sequential()
#        model.add(L.Conv1D(64, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
#        #model.add(L.MaxPooling1D(pool_size=(2, 2)))
#        model.add(L.Conv1D(128, strides=2, kernel_size=4, activation='relu', padding='same'))
#        #model.add(L.MaxPooling1D(pool_size=(2, 2)))
#        model.add(L.Flatten())
#        model.add(L.Dense(256, activation='relu'))
#        model.add(L.Dense(128, activation='relu'))
#        model.add(L.Dense(3, activation='softmax'))
#    
#        # Compile the model
#        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#        model.compile(loss=categorical_crossentropy,
#                      optimizer=opt,
#                      metrics=['accuracy'])
#        return model
        
#class model_3class():
#    def __init__(self, input_shape):
#        self.input_shape = input_shape
#    def model(self):    
#        model = Sequential()
#        model.add(L.Conv1D(64, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
#        model.add(L.Conv1D(128, strides=2, kernel_size=4, activation='relu', padding='same'))
#        model.add(L.Conv1D(256, strides=2, kernel_size=4, activation='relu', padding='same'))
#        model.add(L.Conv1D(512, strides=2, kernel_size=4, activation='relu', padding='same'))
#        model.add(L.Flatten())
#        model.add(L.Dense(1024, activation='relu'))
#        model.add(L.Dropout(0.2))
#        model.add(L.Dense(128, activation='relu'))
#        model.add(L.Dropout(0.2))
#        model.add(L.Dense(3, activation='softmax'))
#    
#        # Compile the model
#        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#        model.compile(loss=categorical_crossentropy,
#                      optimizer=opt,
#                      metrics=['accuracy'])
#        return model
    
class model_3class2D():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
    def model(self):    
        model = Sequential()
        model.add(L.Conv2D(64, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
        model.add(L.Conv2D(128, strides=2, kernel_size=4, activation='relu', padding='same'))
        model.add(L.Conv2D(256, strides=2, kernel_size=4, activation='relu', padding='same'))
        model.add(L.Conv2D(512, strides=2, kernel_size=4, activation='relu', padding='same'))
        model.add(L.Flatten())
        model.add(L.Dense(1024, activation='relu'))
        model.add(L.Dropout(0.2))
        model.add(L.Dense(128, activation='relu'))
        model.add(L.Dropout(0.2))
        model.add(L.Dense(self.num_classes, activation='softmax'))
    
        # Compile the model
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])
        return model
    
#class model_3class2D():
#    def __init__(self, input_shape):
#        self.input_shape = input_shape
#    def model(self):    
#        model = Sequential()
#        model.add(L.Conv2D(32, strides=2, kernel_size=4, activation='relu', padding='same', input_shape=self.input_shape))
#        #model.add(L.MaxPooling1D(pool_size=(2, 2)))
#        model.add(L.Conv2D(64, strides=2, kernel_size=4, activation='relu', padding='same'))
#        #model.add(L.MaxPooling1D(pool_size=(2, 2)))
#        model.add(L.Flatten())
#        model.add(L.Dense(64, activation='relu'))
#        model.add(L.Dense(64, activation='relu'))
#        model.add(L.Dense(3, activation='softmax'))
#    
#        # Compile the model
#        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#        model.compile(loss=categorical_crossentropy,
#                      optimizer=opt,
#                      metrics=['accuracy'])
#        return model



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
        # naming format of the csv: /../<x>sec_<label>.csv
        label = csv_item.split('/')[-1].split('_')[1].replace('.csv', '')
        time = int(csv_item.split('/')[-1].split('_')[0].replace('sec', ''))
        #print('loaded feature shape:', spectrogram.shape, 'label:', label, 'timestamp:', time)
        feat.append(feat_temp)
        labels.append(label)
        timestamp.append(time)
    return feat, labels, timestamp
    
    
#%%
''' load segment features '''

sub_list = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10', 'P11']
target_file_list = ['call', 'reading', 'game', 'TV', 'outdoor', 'dinner'] 

#sub_list = ['P11']
target_file_list = ['call', 'reading', 'game', 'TV', 'outdoor', 'dinner']

downsmaple = False
embedding = True


# obtain features and labels for contextual mfcc features (2D)
features = []
labels = []
groups = []
for sub in sub_list:
    features_sub, labels_sub = [], []
    for ac in target_file_list:
        if embedding:
            seg_dir = '/media/hd4t1/dawei/socialbit/field_study/field_data/%s/seg_30sec_emb/%s/' % (sub, ac)   # dir to load segments
        else:
            seg_dir = '/media/hd4t1/dawei/socialbit/field_study/field_data/%s/seg_30sec/%s/' % (sub, ac)   # dir to load segments
        csv_list = [os.path.join(seg_dir, item) \
                    for item in os.listdir(seg_dir) if item.endswith('.csv')]
        # sort the feature instances by second, naming format of the csv: /../<x>sec_<label>.csv
        csv_list = sorted(csv_list, key=lambda x:int(x.split('/')[-1].split('_')[0].replace('sec', '')))
        feat, labels_ac, timestamp = load_csv(csv_list)   
        print('loaded sub %s class %s' %(sub, ac), len(feat))
        
        # stack data for a sub    
        features_sub.extend(feat)
        labels_sub.extend(labels_ac) 
    
    # resample data for each type (0, 1, 2) to ensure class balance; performed for each sub individually
    if downsmaple:
        # it seems that RandomUnderSampler only takes 2D feat inpt, that's why we need to reshape it
        features_sub = [item.flatten() for item in features_sub]
        #rd = RandomUnderSampler(sampling_strategy='not minority', random_state=0, replacement=False)
        rd = RandomOverSampler(sampling_strategy='not majority', random_state=0)
        features_sub, labels_sub = rd.fit_resample(features_sub, labels_sub)
        # reshape it back
        if embedding:
            features_sub = [np.asarray(item).reshape((1000, -1)) for item in features_sub]
        else:
            features_sub = [np.asarray(item).reshape((48, -1)) for item in features_sub]
        assert len(features_sub) == len(labels_sub), 'resampled feat and labels are not the same size'
        print('Resampled dataset shape {}'.format(Counter(labels_sub)))
    
    # stack data sub by sub    
    features.extend(features_sub)
    labels.extend(labels_sub) 
    groups.extend([sub] * len(labels_sub))
    
features, labels, groups = np.asarray(features), np.asarray(labels), np.asarray(groups)

labels = labels.astype(int) 
     
# standization for features. output: 48D mean and std
#mean = np.mean(features, axis = (0,2))
#std = np.std(features, axis = (0,2))
#for i in range(len(features)):
#    for frame_i in range(features[0].shape[1]):
#        features[i, :, frame_i] = (features[i, :, frame_i] - mean) / std

# data prep for cnn inpt
features = np.expand_dims(features, axis=-1).astype(float)
labels = np.reshape(labels, (len(labels), 1))    
print(features.shape, labels.shape) 

#%%
''' training and validation (NN) '''

predict = True
save_model_path = '/media/hd4t1/dawei/socialbit/field_study/field_data/models_temp/'
check_dirs.check_dir(save_model_path)
batch_size = 128
num_classes = 3
lppo = LeavePGroupsOut(n_groups=1)
#lppo = GroupKFold(n_splits=2)
#kfold = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
fold_no = 1
f1_per_fold, acc_per_fold, pre_per_fold, rec_per_fold, per_class_acc = [], [], [], [], np.empty((0, num_classes))
# training
if not predict:
    #for train, test in kfold.split(features, labels):
    for train, test in lppo.split(features, labels, groups):
        feat_train, labels_train = features[train], labels[train]
        feat_test, labels_test = features[test], labels[test]
        if embedding:
             model_fold = model_3class(input_shape=(1000, 30, 1), num_classes=num_classes).model()
        else:
            model_fold = model_3class2D(input_shape=(48, 600, 1), num_classes=num_classes).model()
          
        # one-hot
        labels_train = tf.keras.utils.to_categorical(labels_train, num_classes=num_classes, dtype=int)
        labels_test = tf.keras.utils.to_categorical(labels_test, num_classes=num_classes, dtype=int)
          
        # Fit data to model, only models with the best val acc (unbalanced) are saved
        model_fold.fit(feat_train, labels_train,
                      batch_size=batch_size,
                      epochs=50,
                      shuffle=True,
                      validation_data=(feat_test, labels_test),
                      verbose=2,
                      callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr = 0.00001),
                                 tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, mode='auto'),
                                 K.callbacks.ModelCheckpoint(save_model_path+"fold%01d_epoch{epoch:02d}_acc{val_accuracy:.4f}.h5" %fold_no, 
                                                             monitor='val_accuracy', 
                                                             verbose=0, 
                                                             save_best_only=True, 
                                                             save_weights_only=True, 
                                                             mode='auto', 
                                                             save_freq='epoch')])
        
        # remove redudant models
        del model_fold
        models = [os.path.join(save_model_path, item) for item in os.listdir(save_model_path) if item.startswith('fold%d' %fold_no)]
        # sort by acc. model path: ../fold<x>_epoch<x>_acc<x>.h5
        models = sorted(models, key=lambda x:float(x.split('/')[-1].split('_')[-1].replace('acc', '').replace('.h5', '')))
        for model in models[:-1]:
            os.remove(model)
              
        fold_no = fold_no + 1
        
# validation                     
if predict:
    models = [os.path.join(save_model_path, item) for item in os.listdir(save_model_path) if item.endswith('.h5')]
    # sort by fold. model path: ../fold<x>_epoch<x>_acc<x>.h5
    models = sorted(models, key=lambda x:int(x.split('/')[-1].split('_')[0].replace('fold', '')))
    
    #for train, test in kfold.split(features, labels):
    fold_no = 1
    for train, test in lppo.split(features, labels, groups=groups):
        feat_train, labels_train = features[train], labels[train]
        feat_test, labels_test = features[test], labels[test]
        if embedding:
             model_pred = model_3class(input_shape=(1000, 30, 1), num_classes=num_classes).model()
        else:
            model_pred = model_3class2D(input_shape=(48, 600, 1), num_classes=num_classes).model()

        labels_test = tf.keras.utils.to_categorical(labels_test, num_classes=num_classes, dtype=int)
        
        model_pred.load_weights(models[fold_no - 1])
        # prediction
        pred_prob = model_pred.predict(feat_test)
        pred = np.argmax(pred_prob, axis=1)
        labels_test = np.argmax(labels_test, axis=1)

        acc_per_fold.append(accuracy_score(labels_test, pred) * 100)
        f1_per_fold.append(f1_score(labels_test, pred, average = 'macro') * 100)
        pre_per_fold.append(precision_score(labels_test, pred, average = 'macro') * 100)
        rec_per_fold.append(recall_score(labels_test, pred, average = 'macro') * 100)  
        
        #Get the confusion matrix
        cm = confusion_matrix(labels_test, pred)
        #Now the normalize the diagonal entries
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #The diagonal entries are the accuracies of each class
        per_class_acc = np.vstack((per_class_acc, cm.diagonal()))
        
        # initialize
        del model_pred
        fold_no = fold_no + 1
        
    f1 = np.mean(f1_per_fold)
    acc = np.mean(acc_per_fold)
    precision = np.mean(pre_per_fold)
    recall = np.mean(rec_per_fold)
    per_class_acc = np.mean(per_class_acc, axis=0)
    print('acc: ', acc, '\n f1: ', f1, 
            '\n precision: ', precision, '\n recall: ', recall,
            '\n per class acc:', per_class_acc)
    
#%%
import matplotlib.pyplot as plt
vis = True
num_classes = 3
save_model_path = '/media/hd4t1/dawei/socialbit/field_study/field_data/models_temp/'
#features = features.reshape((len(features), 1, 1000))
if vis:
    models = [os.path.join(save_model_path, item) for item in os.listdir(save_model_path) if item.endswith('.h5')]
    # sort by fold. model path: ../fold<x>_epoch<x>_acc<x>.h5
    models = sorted(models, key=lambda x:int(x.split('/')[-1].split('_')[0].replace('fold', ''))) 
    #vis_sub = np.where(np.asarray(groups == 'P11'))
    vis_labels = labels
    vis_features = features
    
    if embedding:
        model_pred = model_3class(input_shape=(1000, 30, 1), num_classes=num_classes).model()
    else:
        model_pred = model_3class2D(input_shape=(48, 600, 1), num_classes=num_classes).model()
    vis_labels = tf.keras.utils.to_categorical(vis_labels, num_classes=num_classes, dtype=int)
    fold_no = 11
    model_pred.load_weights(models[fold_no - 1])
    # prediction
    vis_prob = model_pred.predict(vis_features)
    vis_pred = np.argmax(vis_prob, axis=1)
    vis_labels = np.argmax(vis_labels, axis=1)
    
    plt.figure(0)
    # conv [1900:2000], [5700:5850], mono: [0:100], [100:200], back voice: [6500:6650]
    plt.plot(np.arange(len(vis_labels))[:], vis_labels[:], 'o', c='blue', label='true')   
    #plt.figure(1)
    plt.plot(np.arange(len(vis_pred))[:], vis_pred[:]+0.1, 'o', c='coral', label='pred')
    plt.legend()
    plt.xlabel('time frame / sec')
    plt.yticks([0, 1, 2], 
               ['0', '1', '2'])
    plt.show()
    print(f1_score(vis_labels, vis_pred, average = 'macro') * 100)
    print(accuracy_score(vis_labels, vis_pred) * 100)
    #Get the confusion matrix
    vis_cm = confusion_matrix(vis_labels, vis_pred)
    #Now the normalize the diagonal entries
    vis_cm = vis_cm.astype('float') / vis_cm.sum(axis=1)[:, np.newaxis]
    ConfusionMatrixDisplay(vis_cm).plot()  
