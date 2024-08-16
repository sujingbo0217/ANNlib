import argparse
import numpy as np
import os
from collections import defaultdict
from numpy.random import default_rng

class ZipfDistribution:
    def __init__(self, num_points, num_labels, distribution_factor=0.7):
        self.num_labels = num_labels
        self.num_points = num_points
        self.distribution_factor = distribution_factor
        self.rng = default_rng()

    def create_distribution_map(self):
        distribution_map = defaultdict(int)
        primary_label_freq = int(np.ceil(self.num_points * self.distribution_factor))
        for i in range(1, self.num_labels + 1):
            distribution_map[i] = int(np.ceil(primary_label_freq / i))
        return distribution_map

    def generate_distribution(self):
        data = []
        distribution_map = self.create_distribution_map()
        for _ in range(self.num_points):
            labels = []
            for label, count in distribution_map.items():
                if count > 0 and self.rng.random() < self.distribution_factor / label:
                    labels.append(label)
                    distribution_map[label] -= 1
            # if not labels:
            #     labels.append([])
            data.append(labels)
        return data

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic labels.')
    parser.add_argument('-output_file', type=str, required=True, help='Output file path')
    parser.add_argument('-num_points', type=int, required=True, help='Number of points')
    parser.add_argument('-num_labels', type=int, required=True, help='Number of labels')
    parser.add_argument('-distribution_type', type=str, required=True, choices=['zipf', 'random', 'one_per_point'], help='Distribution type')
    args = parser.parse_args()

    num_points = args.num_points
    num_labels = args.num_labels

    print(f'Generating synthetic labels for {num_points} points with {num_labels} unique labels')

    data = []
    if args.distribution_type == 'zipf':
        zipf = ZipfDistribution(num_points, num_labels)
        data = zipf.generate_distribution()
    elif args.distribution_type == 'random':
        rng = default_rng()
        for _ in range(num_points):
            labels = [i + 1 for i in range(num_labels) if rng.random() < 0.5]
            # if not labels:
            #     labels.append([])
            data.append(labels)
    elif args.distribution_type == 'one_per_point':
        rng = default_rng()
        for _ in range(num_points):
            label = rng.integers(1, num_labels + 1)
            data.append([label])

    # Create the labels directory if it doesn't exist
    labels_dir = os.path.dirname(args.output_file)
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    assert(len(data) == num_points)
    total_labels = 0
    separator = np.iinfo(np.uint32).max

    ###
    from collections import defaultdict
    s = defaultdict(list)
    ###
    
    with open(args.output_file, "wb") as f:
        # total points
        # shape = np.array([data_array.shape[0]], dtype=np.uint32)
        # f.write(shape.tobytes())
        f.write(num_points.to_bytes(4, byteorder='little'))
        
        # data
        for i, labels in enumerate(data):
            label_array = np.array(labels, dtype=np.uint32)
            n = label_array.shape[0]
            f.write(label_array.tobytes())
            # separator max
            f.write(separator.to_bytes(4, byteorder='little'))
            total_labels += n
            for label in labels:
                s[label].append(i)

    print(f"{len(data)} points with {total_labels} labels are written to {args.output_file}")
    
    for k, v in s.items():
        print(f"{k}: {len(v)}")

if __name__ == '__main__':
    main()
