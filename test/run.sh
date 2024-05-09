#!/bin/bash

make dyn_test

./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m 32 -efc 128 -b 2 \
    -q ../data/data/sift/sift_query.fvecs:fvecs -ef 100 -k 32 \
    -lb ../data/data/labels/uni_base.txt \
    -lq ../data/data/labels/uni_query.txt \
    1> logs/out_filtered_vamana_uni_k_32.txt \
    2> logs/out_filtered_vamana_uni_k_32.log

./dyn_test -init 20000 -step 20000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m 32 -efc 128 -b 2 \
    -q ../data/data/sift/sift_query.fvecs:fvecs -ef 100 -k 32 \
    -lb ../data/data/labels/labels_zipf_num_1000_base.txt \
    -lq ../data/data/labels/labels_zipf_num_10_query.txt \
    1> logs/out_filtered_vamana_zipf_k_32.txt \
    2> logs/out_filtered_vamana_zipf_k_32.log

./dyn_test -init 20000 -step 20000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m 32 -efc 128 -b 2 \
    -q ../data/data/sift/sift_query.fvecs:fvecs -ef 100 -k 32 \
    -lb ../data/data/labels/labels_random_num_100_base.txt \
    -lq ../data/data/labels/labels_random_num_10_query.txt \
    1> logs/out_filtered_vamana_random_k_32.txt \
    2> logs/out_filtered_vamana_random_k_32.log