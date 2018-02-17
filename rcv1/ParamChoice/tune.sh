#!/bin/sh

set -x

tau_list="1 3 5 10 50 100 1000 5000"

for tau in $tau_list
do
    python sep_tune.py $tau
done
