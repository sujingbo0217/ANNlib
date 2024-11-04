#!/bin/bash

export alpha=0.83
export batch=2
export ml=0.4
export k=10
export m=64
# export efc=128
export efc=100
export data_dir="/localdata/jsu068"

# make filter_test MODE=DEBUG -B &&
make filter_test -B &&

### Random filter label sets

## bigann
./filter_test -init 1000000 -step 1000000 -max 10000000 -type uint8 -dist L2 \
    -R 32 -beam 10,50,100,150,200,250,300,350,400,450,500,550,600,650 \
    -m ${m} -ml ${ml} -efc ${efc} -alpha ${alpha} -b ${batch} -k ${k} \
    -in ${data_dir}/bigann/bigann.128D.10M.euclidean.base.u8bin:u8bin \
    -q ${data_dir}/bigann/bigann.128D.10K.euclidean.query.u8bin:u8bin \
    -gts 1,2,4,10,20,50 \
    -lb ${data_dir}/bigann/bigann.10M.L50.zipf0.75.base.txt \
    -lq ${data_dir}/bigann/bigann.10K.L25.single.query.txt \
    &> "logs/bigann.10M.k${k}.filter.log"

# deep
./filter_test -init 1000000 -step 1000000 -max 10000000 -type float -dist angular \
    -R 32 -beam 10,50,100,150,200,250,300,350,400,450,500,550,600,650 \
    -m ${m} -ml ${ml} -efc ${efc} -alpha ${alpha} -b ${batch} -k ${k} \
    -in ${data_dir}/deep/deep.96D.10M.angular.base.fbin:fbin \
    -q ${data_dir}/deep/deep.96D.10K.angular.query.fbin:fbin \
    -gts 1,2,4,10,20,50 \
    -lb ${data_dir}/deep/deep.10M.L50.zipf0.75.base.txt \
    -lq ${data_dir}/deep/deep.10K.L25.single.query.txt \
    &> "logs/deep.10M.k${k}.filter.log"

### Natural filter label sets

## marco
./filter_test -init 10000000 -step 10000000 -max 0 -type float -dist L2 \
    -R 32 -beam 10,50,100,150,200,250,300,350,400,450,500,550,600,650 \
    -m ${m} -ml ${ml} -efc ${efc} -alpha ${alpha} -b ${batch} -k ${k} \
    -in ${data_dir}/marco/embedding/marco.768D.100M.euclidean.fbin:fbin \
    -q ${data_dir}/marco/query/marco.768D.10K.euclidean.fbin:fbin \
    -gts 50 \
    -lb ${data_dir}/marco/embedding/marco.filter.base.txt \
    -lq ${data_dir}/marco/query/marco.filter.query.txt \
    &> "logs/marco.10M.k${k}.filter.log"

## yfcc
./filter_test -init 1000000 -step 1000000 -max 10000000 -type uint8 -dist L2 \
    -R 32 -beam 10,50,100,150,200,250,300,350,400,450,500,550,600,650 \
    -m ${m} -ml ${ml} -efc ${efc} -alpha ${alpha} -b ${batch} -k ${k} \
    -in ${data_dir}/yfcc/yfcc.192D.10M.euclidean.base.u8bin:u8bin \
    -q ${data_dir}/yfcc/yfcc.192D.100K.euclidean.query.u8bin:u8bin \
    -gts 23,29,89,20,1589,5893 \
    -lb ${data_dir}/yfcc/yfcc.filter.base.txt \
    -lq ${data_dir}/yfcc/yfcc.filter.query.txt \
    &> "logs/yfcc.10M.k${k}.stitched2.log"
