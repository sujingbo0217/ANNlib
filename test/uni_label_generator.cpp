#include <bits/stdc++.h>

using namespace std;

int main() {
  const int N_b = 1000000;
  const int Nq_s = 10000;
  const int Nq_g = 1000;
  ofstream uni_base("../data/labels/uni_base.txt");
  ofstream uni_query_s("../data/labels/uni_query_10k.txt");
  ofstream uni_query_g("../data/labels/uni_query_1k.txt");
  ofstream single_base_12("../data/labels/single_base_12.txt");
  ofstream single_query_12_s("../data/labels/single_query_12_10k.txt");
  ofstream single_query_12_g("../data/labels/single_query_12_1k.txt");
  ofstream single_base_50("../data/labels/single_base_50.txt");
  ofstream single_query_50_s("../data/labels/single_query_50_10k.txt");
  ofstream single_query_50_g("../data/labels/single_query_50_1k.txt");
  ofstream single_base_100("../data/labels/single_base_100.txt");
  ofstream single_query_100_s("../data/labels/single_query_100_10k.txt");
  ofstream single_query_100_g("../data/labels/single_query_100_1k.txt");
  ofstream single_base_500("../data/labels/single_base_500.txt");
  ofstream single_query_500_s("../data/labels/single_query_500_10k.txt");
  ofstream single_query_500_g("../data/labels/single_query_500_1k.txt");
  ofstream single_base_1000("../data/labels/single_base_1000.txt");
  ofstream single_query_1000_s("../data/labels/single_query_1000_10k.txt");
  ofstream single_query_1000_g("../data/labels/single_query_1000_1k.txt");
  mt19937 rng((unsigned int)chrono::steady_clock::now().time_since_epoch().count());
  if (uni_base.is_open() && single_base_12.is_open() && single_base_100.is_open() &&
      single_base_500.is_open() && single_base_1000.is_open() && single_base_50.is_open()) {
    uni_base << N_b << '\n';
    single_base_12 << N_b << '\n';
    single_base_50 << N_b << '\n';
    single_base_100 << N_b << '\n';
    single_base_500 << N_b << '\n';
    single_base_1000 << N_b << '\n';
    for (int i = 0; i < N_b; i++) {
      uni_base << "1" << '\n';
      single_base_12 << rng() % 12 << '\n';
      single_base_50 << rng() % 50 << '\n';
      single_base_100 << rng() % 100 << '\n';
      single_base_500 << rng() % 500 << '\n';
      single_base_1000 << rng() % 1000 << '\n';
    }
    uni_base.close();
    single_base_12.close();
    single_base_50.close();
    single_base_100.close();
    single_base_500.close();
    single_base_1000.close();
  }
  if (uni_query_s.is_open() && single_query_12_s.is_open() && single_query_100_s.is_open() &&
      single_query_500_s.is_open() && single_query_1000_s.is_open() && single_query_50_s.is_open()) {
    uni_query_s << Nq_s << '\n';
    single_query_12_s << Nq_s << '\n';
    single_query_50_s << Nq_s << '\n';
    single_query_100_s << Nq_s << '\n';
    single_query_500_s << Nq_s << '\n';
    single_query_1000_s << Nq_s << '\n';
    for (int i = 0; i < Nq_s; i++) {
      uni_query_s << "1" << '\n';
      single_query_12_s << rng() % 12 << '\n';
      single_query_50_s << rng() % 50 << '\n';
      single_query_100_s << rng() % 100 << '\n';
      single_query_500_s << rng() % 500 << '\n';
      single_query_1000_s << rng() % 1000 << '\n';
    }
    uni_query_s.close();
    single_query_12_s.close();
    single_query_50_s.close();
    single_query_100_s.close();
    single_query_500_s.close();
    single_query_1000_s.close();
  }

  if (uni_query_g.is_open() && single_query_12_g.is_open() && single_query_100_g.is_open() &&
      single_query_500_g.is_open() && single_query_1000_g.is_open() && single_query_50_g.is_open()) {
    uni_query_g << Nq_g << '\n';
    single_query_12_g << Nq_g << '\n';
    single_query_50_g << Nq_g << '\n';
    single_query_100_g << Nq_g << '\n';
    single_query_500_g << Nq_g << '\n';
    single_query_1000_g << Nq_g << '\n';
    for (int i = 0; i < Nq_g; i++) {
      uni_query_g << "1" << '\n';
      single_query_12_g << rng() % 12 << '\n';
      single_query_50_g << rng() % 50 << '\n';
      single_query_100_g << rng() % 100 << '\n';
      single_query_500_g << rng() % 500 << '\n';
      single_query_1000_g << rng() % 1000 << '\n';
    }
    uni_query_g.close();
    single_query_12_g.close();
    single_query_50_g.close();
    single_query_100_g.close();
    single_query_500_g.close();
    single_query_1000_g.close();
  }
  return 0;
}