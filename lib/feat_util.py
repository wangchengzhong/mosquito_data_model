import librosa
import os
import skimage.util
import numpy as np
from tensorflow.python.training.tracking.base import no_manual_dependency_tracking_scope
import config
import pickle
import math
import collections
import pandas as pd
from sklearn.utils import shuffle
import random

def between(l1,low,high):
    l2 = []
    for i in l1:
        if(i>low and i < high):
            l2.append(i)
    return l2
def get_feat(filename,rate,n_feat,is_mosq):
    signal, rate = librosa.load(filename, sr=rate)

    signal = librosa.to_mono(signal)
    if is_mosq:
        for i in range(len(signal)):
            a = 0.1*(random.random()-0.5)
            signal[i] = min(1,signal[i]+a)
            signal[i] = max(signal[i],0)
    feat = librosa.feature.melspectrogram(signal, sr=rate, n_mels=n_feat)            
    feat = librosa.power_to_db(feat, ref=np.max)
    feat = (feat-np.mean(feat))/np.std(feat)
    return feat

def get_feat_train_test(mos_data_dir,env_data_dir, rate, min_duration, n_feat):

    X = []
    y = []
    for filename in os.listdir(mos_data_dir):
        fullname = os.path.join(mos_data_dir,filename)
        length = librosa.get_duration(filename = fullname)
        if length > min_duration:
            feat = get_feat(fullname,rate,n_feat,1)
            X.append(feat)
            y.append(1)
    for filename in os.listdir(env_data_dir):
        fullname = os.path.join(env_data_dir,filename)
        length = librosa.get_duration(filename = fullname)
        if length > min_duration:
            feat = get_feat(fullname,rate,n_feat,0)
            X.append(feat)
            y.append(0)
    return X, y


 



def reshape_feat(feats, labels, win_size, step_size):
    '''Reshaping features from The truth value of an array with more than one element is ambiguous. to be compatible for classifiers expecting a 2D slice as input. Parameter `win_size` is 
    given in number of feature windows (in librosa this is the hop length divided by the sample rate.)
    Can code to be a function of time and hop length instead in future.'''
    
    feats_windowed_array = []
    labels_windowed_array = []
    for idx, feat in enumerate(feats):
        if np.shape(feat)[1] < win_size:
            print('Length of recording shorter than supplied window size.') 
            pass
        else:
            feats_windowed = skimage.util.view_as_windows(feat.T, (win_size,np.shape(feat)[0]), step=step_size)
            labels_windowed = np.full(len(feats_windowed), labels[idx])
            feats_windowed_array.append(feats_windowed)
            labels_windowed_array.append(labels_windowed)
    return np.vstack(feats_windowed_array), np.hstack(labels_windowed_array)



def get_train_test_from_df():
    
    pickle_name_train = 'log_mel_feat_train_'+str(config.n_feat)+'_win_'+str(config.win_size)+'_step_'+str(config.step_size)+'_norm_'+str(config.norm_per_sample)+'.pickle'
     # step = window for test (no augmentation of test):
    pickle_name_test = 'log_mel_feat_test_'+str(config.n_feat)+'_win_'+str(config.win_size)+'_step_'+str(config.win_size)+'_norm_'+str(config.norm_per_sample)+'.pickle'
    
    if not os.path.isfile(os.path.join(config.dir_out_MED, pickle_name_train)):
        print('Extracting training features...')
        X_train, y_train = get_feat_train_test(config.mosq_data_dir,config.env_data_dir,
                                                                     rate=config.rate, min_duration=config.min_duration,
                                                                     n_feat=config.n_feat)
        X_train, y_train = reshape_feat(X_train, y_train, config.win_size, config.step_size)

        log_mel_feat_train = {'X_train':X_train, 'y_train':y_train}#, 'bugs_train':bugs_train}

        
        with open(os.path.join(config.dir_out_MED, pickle_name_train), 'wb') as f:
            pickle.dump(log_mel_feat_train, f, protocol=4)
            print('Saved features to:', os.path.join(config.dir_out_MED, pickle_name_train))

    else:
        print('Loading training features found at:', os.path.join(config.dir_out_MED, pickle_name_train))
        with open(os.path.join(config.dir_out_MED, pickle_name_train), 'rb') as input_file:
            log_mel_feat = pickle.load(input_file)
            X_train = log_mel_feat['X_train']
            y_train = log_mel_feat['y_train']


    return X_train, y_train

