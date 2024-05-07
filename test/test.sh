#!/bin/bash

make dyn_test

./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -ml 0.36 -m 64 -efc 128 -b 2 \
    -q ../data/data/sift/sift_query.fvecs:fvecs -ef 100 -k 32 -idx filtered-hnsw \
    -lb ../data/data/labels/labels_zipf_num_100_base.txt \
    -lq ../data/data/labels/labels_zipf_num_10_query.txt \
    1> logs/out_filtered_hnsw_sift_zipf_100_10_k_32.txt 2> logs/out_filtered_hnsw_sift_zipf_100_10_k_32.log