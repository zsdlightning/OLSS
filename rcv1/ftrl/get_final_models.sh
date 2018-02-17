#!/bin/sh

set -x
#best parameters
ftrl_alpha=1.0
ftrl_beta=0.005
l2_regularization=0.1

l1_list="0 0.001 0.01 0.1 0.5 1 2 3 4 5 6 7 8 9 10 20 30 40 50"
count=0
for l1 in $l1_list
do
        #echo "$version.$count"
        ./run_ftrl2.sh $ftrl_alpha $ftrl_beta $l1 $l2_regularization best.L1.$l1
done
