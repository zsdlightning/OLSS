#!/bin/sh

set -x

version=$1
vw=../../../vw/vw-8.20170116
train=../../../data/news20/news20.binary.shuf.vw.subtrain

$vw -d $train -t -i model.ftrl.$version --invert_hash model.ftrl.$version.inv
count=`cat model.ftrl.$version.inv | wc -l`
count=$((count-11))
echo "feature no. = "$count
