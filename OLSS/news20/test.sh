#!/bin/sh

set -x

model=$1

python predict_offline.py ../../../data/lshtc.test.2  $model > pred
python test_auc.py labels pred

