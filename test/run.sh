#!/bin/bash

# export m=32
# export ef=100
# export efc=128
export alpha=0.83
export batch=2
export ml=0.4
export k=10
export sift=10
export gist=1

# Hyperparameters
M=(32 64)                                                   # R_small R_stitched = R_small + 32
Ef=(90 110 130 150 170 190 210 230 250 270 290 310)         # L (beam size)
Efc=(50 100 150)                                            # L_small
# Alpha=(0.83 0.91 1.00 1.10 1.20)

mkdir -p logs
# rm -rf logs/*
make filter_test -B &&
# make filter_test MODE=DEBUG -B &&

data=("zipf" "uni")
# n_labels=(12 50 100)

for ef in "${Ef[@]}"; do
    for m in "${M[@]}"; do
        for efc in "${Efc[@]}"; do
            for d in "${data[@]}"; do

                ### SIFT-1M
                ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
                    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ml ${ml} -ef ${ef} -efc ${efc} \
                    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -b ${batch} -k ${k} \
                    -lb ../data/labels/base_label_${d}_1M.ubin \
                    -lq ../data/labels/query_label_${d}_${sift}k.ubin \
                    1> logs/sift_${d}_k${k}_m${m}_ef${ef}_efc${efc}.txt \
                    2> logs/sift_${d}_k${k}_m${m}_ef${ef}_efc${efc}.log

                ### GIST-1M
                ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
                    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ml ${ml} -ef ${ef} -efc ${efc}  \
                    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -b ${batch} -k ${k} \
                    -lb ../data/labels/base_label_${d}_1M.ubin \
                    -lq ../data/labels/query_label_${d}_${gist}k.ubin \
                    1> logs/gist_${d}_k${k}_m${m}_ef${ef}_efc${efc}.txt \
                    2> logs/gist_${d}_k${k}_m${m}_ef${ef}_efc${efc}.log

            done
        done
    done
done

### YFCC-1M

#! filtered vamana
# ./filter_test -init 200000 -step 200000 -max 1000000 -type uint8 -dist L2 \
#     -in ../data/data/yfcc100M/base.10M.u8bin.crop_nb_10000000:u8bin -m 96 -ef 100 -efc 330 -b ${batch} \
#     -q ../data/data/yfcc100M/query.public.100K.u8bin:u8bin -alpha 0.9 -k ${k} \
#     -lb ../data/labels/yfcc_base_1M.ubin -lq ../data/labels/yfcc_query_100k.ubin \
#     1> logs/filtered_yfcc1m_k_${k}.txt 2> logs/filtered_yfcc1m_${k}.log

### YFCC-10M

#! filtered vamana
# ./filter_test -init 1000000 -step 1000000 -max 10000000 -type uint8 -dist L2 \
#     -in ../data/data/yfcc100M/base.10M.u8bin.crop_nb_10000000:u8bin -m 96 -ef 100 -efc 330 -b ${batch} \
#     -q ../data/data/yfcc100M/query.public.100K.u8bin:u8bin -alpha 0.9 -k ${k} \
#     -lb ../data/labels/yfcc_base.txt -lq ../data/labels/yfcc_query.txt \
#     1> logs/filtered_yfcc10m_k_${k}.txt 2> logs/filtered_yfcc10m_${k}.log
