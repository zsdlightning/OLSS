#!/bin/sh

set -x

ftrl_alpha=$1
ftrl_beta=$2
version=$3

l2_list="0.1 0.5 1 2 3 4 5"
count=0
for l2 in $l2_list
do
        #echo "$version.$count"
        ./run_ftrl.sh $ftrl_alpha $ftrl_beta 1.0 $l2 $version.$count
        count=$((count+1))
done
