#!/bin/sh

set -x
#this script run the full trainign and test the ture test data
ftrl_alpha=$1
ftrl_beta=$2
l1_regularization=$3
l2_regularization=$4
version=$5
vw=../../../vw/vw-8.20170116
train=../../../data/rcv1/rcv1.train.vw
test=../../../data/rcv1/rcv1.test.vw
labels=./labels
test_auc=../test_auc.py
res=./final.result.txt

$vw -d $train -f model.ftrl.$version  --passes 1 --loss_function logistic --ftrl --ftrl_alpha $ftrl_alpha --ftrl_beta $ftrl_beta --l1 $l1_regularization --l2 $l2_regularization
$vw -d $train -t -i model.ftrl.$version --invert_hash model.ftrl.$version.inv
$vw -d $test -t -i model.ftrl.$version -p p_out.$version
count=`cat model.ftrl.$version.inv | wc -l`
count=$((count-11))
auc=`python $test_auc $labels p_out.$version`
echo $version,L1=$l1_regularization,$auc,feature no.=$count
echo $version,L1=$l1_regularization,$auc,feature no.=$count >> $res
