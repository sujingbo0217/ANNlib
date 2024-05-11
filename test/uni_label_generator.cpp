#include <bits/stdc++.h>

using namespace std;

int main() {
  const int N_b = 1000000;
  const int Nq_s = 10000;
  const int Nq_g = 1000;
  ofstream uni_base("../data/data/labels/uni_base.txt");
  ofstream uni_query_s("../data/data/labels/uni_query_10k.txt");
  ofstream uni_query_g("../data/data/labels/uni_query_1k.txt");
  ofstream single_base_10("../data/data/labels/single_base_10.txt");
  ofstream single_query_10_s("../data/data/labels/single_query_10_10k.txt");
  ofstream single_query_10_g("../data/data/labels/single_query_10_1k.txt");
  ofstream single_base_20("../data/data/labels/single_base_20.txt");
  ofstream single_query_20_s("../data/data/labels/single_query_20_10k.txt");
  ofstream single_query_20_g("../data/data/labels/single_query_20_1k.txt");
  ofstream single_base_30("../data/data/labels/single_base_30.txt");
  ofstream single_query_30_s("../data/data/labels/single_query_30_10k.txt");
  ofstream single_query_30_g("../data/data/labels/single_query_30_1k.txt");
  ofstream single_base_40("../data/data/labels/single_base_40.txt");
  ofstream single_query_40_s("../data/data/labels/single_query_40_10k.txt");
  ofstream single_query_40_g("../data/data/labels/single_query_40_1k.txt");
  ofstream single_base_50("../data/data/labels/single_base_50.txt");
  ofstream single_query_50_s("../data/data/labels/single_query_50_10k.txt");
  ofstream single_query_50_g("../data/data/labels/single_query_50_1k.txt");
  mt19937 rng((unsigned int)chrono::steady_clock::now().time_since_epoch().count());
  if (uni_base.is_open() && single_base_10.is_open() && single_base_20.is_open() &&
      single_base_30.is_open() && single_base_40.is_open() && single_base_50.is_open()) {
    uni_base << N_b << '\n';
    single_base_10 << N_b << '\n';
    single_base_20 << N_b << '\n';
    single_base_30 << N_b << '\n';
    single_base_40 << N_b << '\n';
    single_base_50 << N_b << '\n';
    for (int i = 0; i < N_b; i++) {
      uni_base << "1" << '\n';
      single_base_10 << rng() % 10 << '\n';
      single_base_20 << rng() % 20 << '\n';
      single_base_30 << rng() % 30 << '\n';
      single_base_40 << rng() % 40 << '\n';
      single_base_50 << rng() % 50 << '\n';
    }
    uni_base.close();
    single_base_10.close();
    single_base_20.close();
    single_base_30.close();
    single_base_40.close();
    single_base_50.close();
  }
  if (uni_query_s.is_open() && single_query_10_s.is_open() && single_query_20_s.is_open() &&
      single_query_30_s.is_open() && single_query_40_s.is_open() && single_query_50_s.is_open()) {
    uni_query_s << Nq_s << '\n';
    single_query_10_s << Nq_s << '\n';
    single_query_20_s << Nq_s << '\n';
    single_query_30_s << Nq_s << '\n';
    single_query_40_s << Nq_s << '\n';
    single_query_50_s << Nq_s << '\n';
    for (int i = 0; i < Nq_s; i++) {
      uni_query_s << "1" << '\n';
      single_query_10_s << rng() % 10 << '\n';
      single_query_20_s << rng() % 20 << '\n';
      single_query_30_s << rng() % 30 << '\n';
      single_query_40_s << rng() % 40 << '\n';
      single_query_50_s << rng() % 50 << '\n';
    }
    uni_query_s.close();
    single_query_10_s.close();
    single_query_20_s.close();
    single_query_30_s.close();
    single_query_40_s.close();
    single_query_50_s.close();
  }

  if (uni_query_g.is_open() && single_query_10_g.is_open() && single_query_20_g.is_open() &&
      single_query_30_g.is_open() && single_query_40_g.is_open() && single_query_50_g.is_open()) {
    uni_query_g << Nq_g << '\n';
    single_query_10_g << Nq_g << '\n';
    single_query_20_g << Nq_g << '\n';
    single_query_30_g << Nq_g << '\n';
    single_query_40_g << Nq_g << '\n';
    single_query_50_g << Nq_g << '\n';
    for (int i = 0; i < Nq_g; i++) {
      uni_query_g << "1" << '\n';
      single_query_10_g << rng() % 10 << '\n';
      single_query_20_g << rng() % 20 << '\n';
      single_query_30_g << rng() % 30 << '\n';
      single_query_40_g << rng() % 40 << '\n';
      single_query_50_g << rng() % 50 << '\n';
    }
    uni_query_g.close();
    single_query_10_g.close();
    single_query_20_g.close();
    single_query_30_g.close();
    single_query_40_g.close();
    single_query_50_g.close();
  }
  return 0;
}