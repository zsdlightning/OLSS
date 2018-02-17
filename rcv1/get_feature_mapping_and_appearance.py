#!/bin/python


import os
import sys
import numpy as np

''' input:  VW training file '''
''' output: feature_to_id.txt, feature_appearence_pos.npy, feature_appearence_neg.npy  '''
''' note this is for rcv1, for others with sample weight, we should use items[3:] instead of items[2:] '''


#calculate the appearche of each features in the training data,  for postive and negative samples
#in the end, appending the number of pos. & neg. samples
def calc_per_feature_appearence(d, fea2id, training_file):
    #including intercept
    #pos
    res = np.zeros(d+1)
    #neg
    res2 = np.zeros(d+1)
    with open(training_file, 'r') as f:
        ct = 0
        for line in f:
            ct = ct + 1
            items = line.strip().split(' ')
            label = int(items[0])
            if label  == 1:
                res[d] = res[d] + 1
            else:
                res2[d] = res2[d] + 1

            for item in items[2:]:
                name = item.split(':')[0]
                id = fea2id[name]
                if label == 1:
                    res[ id ] = res[ id ] + 1
                else:
                    res2[ id ] = res2[ id ] + 1
            if ct%10000 == 0:
                print ct
    np.save('feature_appearence_pos.npy',res)
    np.save('feature_appearence_neg.npy',res2)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s <VW training file>'%sys.argv[0]
        sys.exit(1)

    #VW format
    features = {}
    ct = 0
    with open(sys.argv[1],'r') as f:
        for line in f:
            items = line.strip().split(' ')
            ct = ct + 1
            if ct%10000 == 0:
                print ct
            for item in items[2:]:
                key = item.split(':')[0]
                if key not in features:
                    id = len(features)
                    features[ key ] = id

    with open('feature_to_id.txt', 'w') as f:
        for key in features:
            id = features[key]
            print >>f,  '%s\t%d'%(key, id)

    calc_per_feature_appearence(len(features), features, sys.argv[1])

