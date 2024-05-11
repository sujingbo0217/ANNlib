mkdir ../data/data/labels
make label_generator -B
g++ -o uni_label_generator uni_label_generator.cpp

./label_generator -output_file ../data/data/labels/zipf_10_base.txt -num_points 1000000 -num_labels 10 -distribution_type zipf
./label_generator -output_file ../data/data/labels/zipf_50_base.txt -num_points 1000000 -num_labels 50 -distribution_type zipf
./label_generator -output_file ../data/data/labels/zipf_100_base.txt -num_points 1000000 -num_labels 100 -distribution_type zipf

./label_generator -output_file ../data/data/labels/zipf_10_query_10k.txt -num_points 10000 -num_labels 10 -distribution_type zipf
./label_generator -output_file ../data/data/labels/zipf_50_query_10k.txt -num_points 10000 -num_labels 50 -distribution_type zipf
./label_generator -output_file ../data/data/labels/zipf_100_query_10k.txt -num_points 10000 -num_labels 100 -distribution_type zipf

./label_generator -output_file ../data/data/labels/zipf_10_query_1k.txt -num_points 1000 -num_labels 10 -distribution_type zipf
./label_generator -output_file ../data/data/labels/zipf_50_query_1k.txt -num_points 1000 -num_labels 50 -distribution_type zipf
./label_generator -output_file ../data/data/labels/zipf_100_query_1k.txt -num_points 1000 -num_labels 100 -distribution_type zipf


./label_generator -output_file ../data/data/labels/random_10_base.txt -num_points 1000000 -num_labels 10 -distribution_type random
./label_generator -output_file ../data/data/labels/random_50_base.txt -num_points 1000000 -num_labels 50 -distribution_type random
./label_generator -output_file ../data/data/labels/random_100_base.txt -num_points 1000000 -num_labels 100 -distribution_type random

./label_generator -output_file ../data/data/labels/random_10_query_10k.txt -num_points 10000 -num_labels 10 -distribution_type random
./label_generator -output_file ../data/data/labels/random_50_query_10k.txt -num_points 10000 -num_labels 50 -distribution_type random
./label_generator -output_file ../data/data/labels/random_100_query_10k.txt -num_points 10000 -num_labels 100 -distribution_type random

./label_generator -output_file ../data/data/labels/random_10_query_1k.txt -num_points 1000 -num_labels 10 -distribution_type random
./label_generator -output_file ../data/data/labels/random_50_query_1k.txt -num_points 1000 -num_labels 50 -distribution_type random
./label_generator -output_file ../data/data/labels/random_100_query_1k.txt -num_points 1000 -num_labels 100 -distribution_type random

./uni_label_generator