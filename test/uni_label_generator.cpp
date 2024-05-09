#include <bits/stdc++.h>

using namespace std;

int main() {
  const int N_b = 1000000;
  const int N_q = 10000;
  ofstream output_base("../data/data/labels/uni_base.txt");
  ofstream output_query("../data/data/labels/uni_query.txt");
  if (output_base.is_open()) {
    for (int i = 0; i < N_b; i++) {
      output_base << "1" << '\n';
    }
    output_base.close();
  }
  if (output_query.is_open()) {
    for (int i = 0; i < N_q; i++) {
      output_query << "1" << '\n';
    }
    output_query.close();
  }
  return 0;
}