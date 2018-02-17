from sklearn import metrics

import sys
import os

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print '%s <label> <pred>'%sys.argv[0]
        sys.exit(0)

    y = []
    with open(sys.argv[1], 'r') as f:
        for line in f:
            y.append(int(line))
    pred = []
    pred_label = []
    with open(sys.argv[2], 'r') as f:
        for line in f:
            val = float(line)
            pred.append(val)
            if val<0:
                pred_label.append(-1)
            else:
                pred_label.append(1)
            


    fpr,tpr,th = metrics.roc_curve(y, pred, pos_label=1)
    val = metrics.auc(fpr,tpr)
    f1 = metrics.f1_score(y, pred_label, pos_label=1)
    print 'auc = ', val,  ', f1 = ', f1
