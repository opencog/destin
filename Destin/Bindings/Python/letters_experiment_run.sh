#!/bin/bash

log=letters_experiment.log

experiment_id=1
experiment_skip_id=0

freq_coeffs="0.001 0.01 0.05 0.1 0.2 0.5"
freq_tresholds="0.000001 0.00001 0.0001 0.001 0.01, 0.1"
add_coeffs="0.1 0.5 1 5 10"
starv_coeff="0"

cd "$(dirname "$0")"

for freq_coeff in $freq_coeffs; do
  for freq_treshold in $freq_tresholds; do
    for add_coeff in $add_coeffs; do

      if [ $experiment_id -le $experiment_skip_id ]; then
        (( experiment_id++ ))
        continue;
      fi

      cmd="python letters_experiment.py $experiment_id $freq_coeff $freq_treshold $add_coeff $starv_coeff 2 50"
      echo "STARTING NEW EXPERIMENT ...... " | tee -a $log
      echo $cmd | tee -a $log
      $cmd | tee -a $log
      (( experiment_id++ ))

      done
  done
done
