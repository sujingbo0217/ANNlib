#!/bin/bash

export m=32
export ef=100
export efc=128
export batch=2
export alpha=0.83
export k=10

mkdir logs && rm -rf logs/*
make dyn_test -B &&

### SIFT
# Uni
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/uni_base.txt \
    -lq ../data/labels/uni_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_uni.txt \
    2> logs/filtered_vamana_sift_k_10_uni.log \
&&
# Single
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/single_base_10.txt \
    -lq ../data/labels/single_query_10_10k.txt \
    1> logs/filtered_vamana_sift_k_10_single_10.txt \
    2> logs/filtered_vamana_sift_k_10_single_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/single_base_20.txt \
    -lq ../data/labels/single_query_20_10k.txt \
    1> logs/filtered_vamana_sift_k_10_single_20.txt \
    2> logs/filtered_vamana_sift_k_10_single_20.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/single_base_30.txt \
    -lq ../data/labels/single_query_30_10k.txt \
    1> logs/filtered_vamana_sift_k_10_single_30.txt \
    2> logs/filtered_vamana_sift_k_10_single_30.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/single_base_40.txt \
    -lq ../data/labels/single_query_40_10k.txt \
    1> logs/filtered_vamana_sift_k_10_single_40.txt \
    2> logs/filtered_vamana_sift_k_10_single_40.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/single_base_50.txt \
    -lq ../data/labels/single_query_50_10k.txt \
    1> logs/filtered_vamana_sift_k_10_single_50.txt \
    2> logs/filtered_vamana_sift_k_10_single_50.log \
&&
# ZIPF
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_10_base.txt \
    -lq ../data/labels/zipf_10_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_zipf_10_10.txt \
    2> logs/filtered_vamana_sift_k_10_zipf_10_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_50_base.txt \
    -lq ../data/labels/zipf_10_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_zipf_50_10.txt \
    2> logs/filtered_vamana_sift_k_10_zipf_50_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_50_base.txt \
    -lq ../data/labels/zipf_50_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_zipf_50_50.txt \
    2> logs/filtered_vamana_sift_k_10_zipf_50_50.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_100_base.txt \
    -lq ../data/labels/zipf_10_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_zipf_100_10.txt \
    2> logs/filtered_vamana_sift_k_10_zipf_100_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_100_base.txt \
    -lq ../data/labels/zipf_50_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_zipf_100_50.txt \
    2> logs/filtered_vamana_sift_k_10_zipf_100_50.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_100_base.txt \
    -lq ../data/labels/zipf_100_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_zipf_100_100.txt \
    2> logs/filtered_vamana_sift_k_10_zipf_100_100.log \
&&
# Random
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_10_base.txt \
    -lq ../data/labels/random_10_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_random_10_10.txt \
    2> logs/filtered_vamana_sift_k_10_random_10_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_50_base.txt \
    -lq ../data/labels/random_10_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_random_50_10.txt \
    2> logs/filtered_vamana_sift_k_10_random_50_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_50_base.txt \
    -lq ../data/labels/random_50_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_random_50_50.txt \
    2> logs/filtered_vamana_sift_k_10_random_50_50.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_100_base.txt \
    -lq ../data/labels/random_10_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_random_100_10.txt \
    2> logs/filtered_vamana_sift_k_10_random_100_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_100_base.txt \
    -lq ../data/labels/random_50_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_random_100_50.txt \
    2> logs/filtered_vamana_sift_k_10_random_100_50.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_100_base.txt \
    -lq ../data/labels/random_100_query_10k.txt \
    1> logs/filtered_vamana_sift_k_10_random_100_100.txt \
    2> logs/filtered_vamana_sift_k_10_random_100_100.log \
&&

## GIST
# Uni
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/uni_base.txt \
    -lq ../data/labels/uni_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_uni.txt \
    2> logs/filtered_vamana_gist_k_10_uni.log \
&&
# Single
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/single_base_10.txt \
    -lq ../data/labels/single_query_10_1k.txt \
    1> logs/filtered_vamana_gist_k_10_single_10.txt \
    2> logs/filtered_vamana_gist_k_10_single_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/single_base_20.txt \
    -lq ../data/labels/single_query_20_1k.txt \
    1> logs/filtered_vamana_gist_k_10_single_20.txt \
    2> logs/filtered_vamana_gist_k_10_single_20.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/single_base_30.txt \
    -lq ../data/labels/single_query_30_1k.txt \
    1> logs/filtered_vamana_gist_k_10_single_30.txt \
    2> logs/filtered_vamana_gist_k_10_single_30.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/single_base_40.txt \
    -lq ../data/labels/single_query_40_1k.txt \
    1> logs/filtered_vamana_gist_k_10_single_40.txt \
    2> logs/filtered_vamana_gist_k_10_single_40.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/single_base_50.txt \
    -lq ../data/labels/single_query_50_1k.txt \
    1> logs/filtered_vamana_gist_k_10_single_50.txt \
    2> logs/filtered_vamana_gist_k_10_single_50.log \
&&
# ZIPF
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_10_base.txt \
    -lq ../data/labels/zipf_10_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_zipf_10_10.txt \
    2> logs/filtered_vamana_gist_k_10_zipf_10_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_50_base.txt \
    -lq ../data/labels/zipf_10_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_zipf_50_10.txt \
    2> logs/filtered_vamana_gist_k_10_zipf_50_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_50_base.txt \
    -lq ../data/labels/zipf_50_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_zipf_50_50.txt \
    2> logs/filtered_vamana_gist_k_10_zipf_50_50.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_100_base.txt \
    -lq ../data/labels/zipf_10_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_zipf_100_10.txt \
    2> logs/filtered_vamana_gist_k_10_zipf_100_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_100_base.txt \
    -lq ../data/labels/zipf_50_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_zipf_100_50.txt \
    2> logs/filtered_vamana_gist_k_10_zipf_100_50.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/zipf_100_base.txt \
    -lq ../data/labels/zipf_100_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_zipf_100_100.txt \
    2> logs/filtered_vamana_gist_k_10_zipf_100_100.log \
&&
# Random
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_10_base.txt \
    -lq ../data/labels/random_10_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_random_10_10.txt \
    2> logs/filtered_vamana_gist_k_10_random_10_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_50_base.txt \
    -lq ../data/labels/random_10_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_random_50_10.txt \
    2> logs/filtered_vamana_gist_k_10_random_50_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_50_base.txt \
    -lq ../data/labels/random_50_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_random_50_50.txt \
    2> logs/filtered_vamana_gist_k_10_random_50_50.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_100_base.txt \
    -lq ../data/labels/random_10_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_random_100_10.txt \
    2> logs/filtered_vamana_gist_k_10_random_100_10.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_100_base.txt \
    -lq ../data/labels/random_50_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_random_100_50.txt \
    2> logs/filtered_vamana_gist_k_10_random_100_50.log \
&&
./dyn_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
    -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
    -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} \
    -lb ../data/labels/random_100_base.txt \
    -lq ../data/labels/random_100_query_1k.txt \
    1> logs/filtered_vamana_gist_k_10_random_100_100.txt \
    2> logs/filtered_vamana_gist_k_10_random_100_100.log