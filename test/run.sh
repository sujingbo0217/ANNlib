#!/bin/bash

# export m=32
# export ef=100
# export efc=128
export alpha=0.83
export batch=2
export k=10
export sift=10
export gist=1

# Hyperparameters 2*12*3*5
M=(32 64)                                                   # R_small R_stitched = R_small + 32
Ef=(90 110 130 150 170 190 210 230 250 270 290 310)         # L (beam size)
Efc=(50 100 150)                                            # L_small
# Alpha=(0.83 0.91 1.00 1.10 1.20)

# mkdir -p logs
# rm -rf logs/*
make filter_test -B

data=("uni" "zipf")
n_labels=(12 50 100)

export d="zipf"
export n=12

for m in "${M[@]}"; do
    for ef in "${Ef[@]}"; do
        for efc in "${Efc[@]}"; do
            ### SIFT-1M

            # vamana
            ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
                -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
                -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} -vt vamana \
                1> logs/vamana_sift_k_${k}_m${m}_ef${ef}_efc${efc}.txt \
                2> logs/vamana_sift_k_${k}_m${m}_ef${ef}_efc${efc}.log

            for d in "${data[@]}"; do
                for n in "${n_labels[@]}"; do

                    if [[ "$d" == "zipf" && "$n" == 100 ]]; then
                        continue
                    fi

                    # filtered vamana
                    ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
                        -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
                        -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} -vt filtered_vamana \
                        -lb ../data/labels/${d}_${n}_base.txt -lq ../data/labels/${d}_${n}_query_${sift}k.txt \
                        1> logs/filtered_sift_k${k}_${d}${n}_m${m}_ef${ef}_efc${efc}.txt \
                        2> logs/filtered_sift_k${k}_${d}${n}_m${m}_ef${ef}_efc${efc}.log

                    # stitched vamana
                    ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
                        -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
                        -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} -vt stitched_vamana \
                        -lb ../data/labels/${d}_${n}_base.txt -lq ../data/labels/${d}_${n}_query_${sift}k.txt \
                        1> logs/stitched_sift_k${k}_${d}${n}_m${m}_ef${ef}_efc${efc}.txt \
                        2> logs/stitched_sift_k${k}_${d}${n}_m${m}_ef${ef}_efc${efc}.log
                done
            done

            ### GIST-1M

            # # vamana
            # ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
            #     -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
            #     -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} -vt vamana \
            #     1> logs/vamana_gist_k_${k}.txt 2> logs/vamana_gist_k_${k}.log

            # for d in "${data[@]}"; do
            #     for n in "${n_labels[@]}"; do
            #         if [[ "$d" == "zipf" && "$n" == 1 ]]; then
            #             continue
            #         fi

            #         # filtered vamana
            #         ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
            #             -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
            #             -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} -vt filtered_vamana \
            #             -lb ../data/labels/${d}_${n}_base.txt -lq ../data/labels/${d}_${n}_query_${gist}k.txt \
            #             1> logs/filtered_gist_k_${k}_${d}_${n}.txt 2> logs/filtered_gist_k_${k}_${d}_${n}.log

            #         # stitched vamana
            #         ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
            #             -in ../data/data/gist/gist_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
            #             -q ../data/data/gist/gist_query.fvecs:fvecs -alpha ${alpha} -k ${k} -vt stitched_vamana \
            #             -lb ../data/labels/${d}_${n}_base.txt -lq ../data/labels/${d}_${n}_query_${gist}k.txt \
            #             1> logs/stitched_gist_k${k}_${d}${n}_m${m}_ef${ef}_efc${efc}_alpha${alpha}.txt \
            #             2> logs/stitched_gist_k${k}_${d}${n}_m${m}_ef${ef}_efc${efc}_alpha${alpha}.log
            #     done
            # done

            ## MS-TURING

            # # vamana
            # ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
            #     -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
            #     -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} -vt vamana \
            #     1> logs/vamana_sift_k_${k}.txt 2> logs/vamana_sift_k_${k}.log

            # for d in "${data[@]}"; do
            #     for n in "${n_labels[@]}"; do
            #         if [[ "$d" == "zipf" && "$n" == 1 ]]; then
            #             continue
            #         fi

            #         # filtered vamana
            #         ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
            #             -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
            #             -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} -vt filtered_vamana \
            #             -lb ../data/labels/${d}_${n}_base.txt -lq ../data/labels/${d}_${n}_query_${sift}k.txt \
            #             1> logs/filtered_sift_k_${k}_${d}_${n}.txt 2> logs/filtered_sift_k_${k}_${d}_${n}.log

            #         # stitched vamana
            #         ./filter_test -init 200000 -step 200000 -max 1000000 -type float -dist L2 \
            #             -in ../data/data/sift/sift_base.fvecs:fvecs -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
            #             -q ../data/data/sift/sift_query.fvecs:fvecs -alpha ${alpha} -k ${k} -vt stitched_vamana \
            #             -lb ../data/labels/${d}_${n}_base.txt -lq ../data/labels/${d}_${n}_query_${sift}k.txt \
            #             1> logs/stitched_sift_k${k}_${d}${n}_m${m}_ef${ef}_efc${efc}_alpha${alpha}.txt \
            #             2> logs/stitched_sift_k${k}_${d}${n}_m${m}_ef${ef}_efc${efc}_alpha${alpha}.log
            #     done
            # done
        done
    done
done

### YFCC-1M

# # vamana
# ./filter_test -init 200000 -step 200000 -max 1000000 -type uint8 -dist L2 \
#     -in ../data/data/yfcc100M/base.10M.u8bin.crop_nb_10000000:u8bin -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
#     -q ../data/data/yfcc100M/query.public.100K.u8bin:u8bin -alpha ${alpha} -k ${k} -vt vamana \
#     1> logs/vamana_yfcc1m_k_${k}.txt 2> logs/vamana_yfcc1m_${k}.log

#! filtered vamana
# ./filter_test -init 200000 -step 200000 -max 1000000 -type uint8 -dist L2 \
#     -in ../data/data/yfcc100M/base.10M.u8bin.crop_nb_10000000:u8bin -m 96 -ef 100 -efc 330 -b ${batch} \
#     -q ../data/data/yfcc100M/query.public.100K.u8bin:u8bin -alpha 0.9 -k ${k} -vt filtered_vamana \
#     -lb ../data/labels/yfcc_base.txt -lq ../data/labels/yfcc_query.txt \
#     1> logs/filtered_yfcc1m_k_${k}.txt 2> logs/filtered_yfcc1m_${k}.log

# stitched vamana
# ./filter_test -init 200000 -step 200000 -max 1000000 -type uint8 -dist L2 \
#     -in ../data/data/yfcc100M/base.10M.u8bin.crop_nb_10000000:u8bin -m ${m} -ef ${ef} -efc ${efc} -b ${batch} \
#     -q ../data/data/yfcc100M/query.public.100K.u8bin:u8bin -alpha ${alpha} -k ${k} -vt stitched_vamana \
#     -lb ../data/labels/yfcc_base.txt -lq ../data/labels/yfcc_query.txt \
#     1> logs/stitched_yfcc1m_k_${k}.txt 2> logs/stitched_yfcc1m_${k}.log

### YFCC-10M

# # vamana
# ./filter_test -init 1000000 -step 1000000 -max 10000000 -type uint8 -dist L2 \
#     -in ../data/data/yfcc100M/base.10M.u8bin.crop_nb_10000000:u8bin -m $((m * 2)) -ef $((ef * 2)) -efc $((efc * 2)) -b ${batch} \
#     -q ../data/data/yfcc100M/query.public.100K.u8bin:u8bin -alpha ${alpha} -k ${k} -vt vamana \
#     1> logs/vamana_yfcc10m_k_${k}.txt 2> logs/vamana_yfcc10m_${k}.log

#! filtered vamana
# ./filter_test -init 1000000 -step 1000000 -max 10000000 -type uint8 -dist L2 \
#     -in ../data/data/yfcc100M/base.10M.u8bin.crop_nb_10000000:u8bin -m 96 -ef 100 -efc 330 -b ${batch} \
#     -q ../data/data/yfcc100M/query.public.100K.u8bin:u8bin -alpha 0.9 -k ${k} -vt filtered_vamana \
#     -lb ../data/labels/yfcc_base.txt -lq ../data/labels/yfcc_query.txt \
#     1> logs/filtered_yfcc10m_k_${k}.txt 2> logs/filtered_yfcc10m_${k}.log

# stitched vamana
# ./filter_test -init 1000000 -step 1000000 -max 10000000 -type uint8 -dist L2 \
#     -in ../data/data/yfcc100M/base.10M.u8bin.crop_nb_10000000:u8bin -m $((m * 2)) -ef $((ef * 2)) -efc $((efc * 2)) -b ${batch} \
#     -q ../data/data/yfcc100M/query.public.100K.u8bin:u8bin -alpha ${alpha} -k ${k} -vt stitched_vamana \
#     -lb ../data/labels/yfcc_base.txt -lq ../data/labels/yfcc_query.txt \
#     1> logs/stitched_yfcc10m_k_${k}.txt 2> logs/stitched_yfcc10m_${k}.log
