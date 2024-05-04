#!/bin/bash

mkdir ../data/data/labels

# ZIPF
./label_generator -output_file ../data/data/labels/labels_zipf_num_100_base.txt -num_points 1000000 -num_labels 100 -distribution_type zipf
./label_generator -output_file ../data/data/labels/labels_zipf_num_1000_base.txt -num_points 1000000 -num_labels 1000 -distribution_type zipf

./label_generator -output_file ../data/data/labels/labels_zipf_num_10_query.txt -num_points 10000 -num_labels 10 -distribution_type zipf
./label_generator -output_file ../data/data/labels/labels_zipf_num_100_query.txt -num_points 10000 -num_labels 100 -distribution_type zipf

# RANDOM
./label_generator -output_file ../data/data/labels/labels_random_num_100_base.txt -num_points 1000000 -num_labels 100 -distribution_type random

./label_generator -output_file ../data/data/labels/labels_random_num_10_query.txt -num_points 1000 -num_labels 10 -distribution_type random
./label_generator -output_file ../data/data/labels/labels_random_num_100_query.txt -num_points 1000 -num_labels 100 -distribution_type random