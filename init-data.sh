#!/bin/bash

git clone https://github.com/laiguokun/multivariate-time-series-data.git ./raw-data

DIR="./raw-data"
OUTPUT_DIR="./dataset"

for dataset in electricity exchange_rate traffic
do
    gzip $DIR/$dataset/$dataset.txt.gz -d
    python split_dataset.py $DIR/$dataset/$dataset.txt $OUTPUT_DIR
done

gzip $DIR/solar-energy/solar_AL.txt.gz -d
python split_dataset.py $DIR/solar-energy/solar_AL.txt $OUTPUT_DIR --output_name solar
