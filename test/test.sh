#!/bin/bash

# make dyn_test

./dyn_test -init 2000 -step 2000 -max 10000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m 32 -efc 128 -b 2 \
    -q ../data/data/sift/sift_query.fvecs:fvecs -ef 100 -k 32 \
    -lb ../data/data/labels/uni_base_1000000.txt -lq ../data/data/labels/uni_query_10000.txt \
    1> logs/out_filtered_vamana_uni_k_32.txt \
    2> logs/out_filtered_vamana_uni_k_32.log