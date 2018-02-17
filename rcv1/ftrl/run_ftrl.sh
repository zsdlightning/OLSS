#!/bin/sh

set -x

ftrl_alpha=$1
ftrl_beta=$2
l1_regularization=$3
l2_regularization=$4
version=$5
vw=../../../vw/vw-8.20170116
train=../../../data/rcv1/rcv1.train.vw.subtrain
test=../../../data/rcv1/rcv1.train.vw.validation
labels=./labels.validation
test_auc=../test_auc.py
res=./tune.result.txt

$vw -d $train -f model.ftrl.$version  --passes 1 --loss_function logistic --ftrl --ftrl_alpha $ftrl_alpha --ftrl_beta $ftrl_beta --l1 $l1_regularization --l2 $l2_regularization
$vw -d $train -t -i model.ftrl.$version --invert_hash model.ftrl.$version.inv
$vw -d $test -t -i model.ftrl.$version -p p_out.$version

auc=`python $test_auc $labels p_out.$version`
echo $version",alpha="$ftrl_alpha",beta="$ftrl_beta,$auc
echo $version",alpha="$ftrl_alpha",beta="$ftrl_beta,$auc >> $res

