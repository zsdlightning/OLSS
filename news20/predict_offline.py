#!/bin/python
import os
import sys
import numpy as np

'''
load model, distribute predict every test samples
'''

feature_off = 3

def load_feature_id(file_path):
    res = {}
    with open(file_path, 'r') as f:
        for line in f:
            name,id = line.strip().split('\t')
            res[ name ] = int(id)
    return res

def load_feature_stats(file_path):
    res = {}
    with open(file_path, 'r') as f:
        for line in f:
            name,mean,std= line.strip().split('\t')
            res[ name ] = [float(mean), float(std)]
    return res



def normalize(val, mean_std):
    return (val - mean_std[0])/mean_std[1]

def predict_one(line, w, fea2id, fea2stats):
    items = line.strip().split(' ')
    label = items[0]
    pred = w[-1]

    for item in items[feature_off:]:
        key_val = item.split(':')
        if key_val[0] not in fea2id:
            continue
        id = fea2id[ key_val[0] ]
        if len(key_val) == 1:
            pred += w[id]
        else:
            pred += (w[id]*normalize(float(key_val[1]), fea2stats[key_val[0]])) 
    return pred
                
if __name__ == '__main__':

    if len(sys.argv) != 3:
        print 'Usage: %s <test-file> <model.npy>'%sys.argv[0]
        sys.exit(0)

    w = np.load(sys.argv[2])
    output = '%s.pred'%(sys.argv[2])
    fea2id = load_feature_id('feature_to_id.txt')
    fea2stats = load_feature_stats('mean_std_continuous_features.txt')
    with open(sys.argv[1], 'r') as f:
        for line in f:
            print predict_one(line, w, fea2id, fea2stats)
