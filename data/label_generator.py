import argparse
import math
import random
from collections import defaultdict

class ZipfDistribution:
    def __init__(self, num_points: int, num_labels: int, distribution_factor: float = 0.7):
        self.num_labels = num_labels
        self.num_points = num_points
        self.distribution_factor = distribution_factor
        self.rand_engine = random.Random()

    def create_distribution_map(self):
        distribution_map = {}
        primary_label_freq = math.ceil(self.num_points * self.distribution_factor)
        for i in range(1, self.num_labels + 1):
            distribution_map[i] = math.ceil(primary_label_freq / i)
        return distribution_map

    def write_distribution(self, outfile):
        distribution_map = self.create_distribution_map()
        for i in range(self.num_points):
            label_written = False
            for label, count in distribution_map.items():
                label_selection_probability = self.distribution_factor / label
                if self.rand_engine.random() < label_selection_probability and distribution_map[label] > 0:
                    if label_written:
                        outfile.write(',')
                    outfile.write(str(label))
                    label_written = True
                    distribution_map[label] -= 1
            if not label_written:
                outfile.write('0')
            if i < self.num_points - 1:
                outfile.write('\n')

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic label distribution for dataset points")
    parser.add_argument("-output_file", "--output_file", required=True, help="Filename for saving the label file")
    parser.add_argument("-num_points", "--num_points", required=True, type=int, help="Number of points in dataset")
    parser.add_argument("-num_labels", "--num_labels", required=True, type=int, help="Number of unique labels, up to 5000")
    parser.add_argument("-distribution_type", "--distribution_type", default="random", help="Distribution type (random/zipf/one_per_point)")
    parser.add_argument("-distribution_factor", "--distribution_factor", default=0.7, type=float, help="Distribution factor for zipf distribution")

    args = parser.parse_args()

    if args.num_labels > 5000:
        print("Error: num_labels must be 5000 or less")
        return -1
    if args.num_points <= 0:
        print("Error: num_points must be greater than 0")
        return -1

    print(f"Generating synthetic labels for {args.num_points} points with {args.num_labels} unique labels")

    try:
        with open(args.output_file, 'w') as outfile:
            if args.distribution_type == "zipf":
                zipf = ZipfDistribution(args.num_points, args.num_labels, args.distribution_factor)
                zipf.write_distribution(outfile)
            elif args.distribution_type == "random":
                for i in range(args.num_points):
                    label_written = False
                    for j in range(1, args.num_labels + 1):
                        if random.random() > 0.5:
                            if label_written:
                                outfile.write(',')
                            outfile.write(str(j))
                            label_written = True
                    if not label_written:
                        outfile.write('0')
                    if i < args.num_points - 1:
                        outfile.write('\n')
            elif args.distribution_type == "one_per_point":
                for i in range(args.num_points):
                    outfile.write(str(random.randint(1, args.num_labels)))
                    if i < args.num_points - 1:
                        outfile.write('\n')

        print(f"Labels written to {args.output_file}")
    except Exception as ex:
        print(f"Label generation failed: {ex}")
        return -1

if __name__ == "__main__":
    main()
