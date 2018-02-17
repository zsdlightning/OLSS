#!/bin/sh

set -x

#rm -f model.ftrl.*

alpha_list="0.005 0.0075 0.01 0.025 0.05 0.075 0.1 0.25 0.5 0.75 1"
beta_list="0.005 0.0075 0.01 0.025 0.05 0.075 0.1 0.25 0.5 0.75 1"
count=0
for alpha in $alpha_list
do
    for  beta in $beta_list
    do
        #echo "$count,$alpha,$beta"
        ./run_ftrl.sh $alpha $beta 1.0 1.0 $count
        count=$((count+1))
    done
done

