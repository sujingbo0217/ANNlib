#!/bin/bash

# mkdir ../data/data/labels
# make label_generator
./label_generator -output_file ../data/data/labels/labels_zipf_num_1000_base.txt -num_points 1000000 -num_labels 100 -distribution_type zipf
./label_generator -output_file ../data/data/labels/labels_zipf_num_100_query.txt -num_points 10000 -num_labels 100 -distribution_type zipf
./label_generator -output_file ../data/data/labels/labels_random_num_100_base.txt -num_points 1000000 -num_labels 100 -distribution_type random
./label_generator -output_file ../data/data/labels/labels_random_num_100_query.txt -num_points 10000 -num_labels 100 -distribution_type random
./uni_label_generator

rm -rf dyn_test && make dyn_test

./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m 64 -ef 100 -efc 100 -b 2 \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha 1.2 -k 32 \
    -lb ../data/data/labels/uni_base.txt \
    -lq ../data/data/labels/uni_query.txt \
    1> logs/out_filtered_vamana_uni_k_32_uni.txt \
    2> logs/out_filtered_vamana_uni_k_32_uni.log

./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m 64 -ef 100 -efc 100 -b 2 \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha 1.2 -k 32 \
    -lb ../data/data/labels/labels_zipf_num_1000_base.txt \
    -lq ../data/data/labels/labels_zipf_num_100_query.txt \
    1> logs/out_filtered_vamana_zipf_k_32_1000_100.txt \
    2> logs/out_filtered_vamana_zipf_k_32_1000_100.log

./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m 64 -ef 100 -efc 100 -b 2 \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha 1.2 -k 32 \
    -lb ../data/data/labels/labels_random_num_100_base.txt \
    -lq ../data/data/labels/labels_random_num_100_query.txt \
    1> logs/out_filtered_vamana_random_k_32_100_100.txt \
    2> logs/out_filtered_vamana_random_k_32_100_100.log