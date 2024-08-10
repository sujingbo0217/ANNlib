#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "../utils.hpp"
#include "algo/HNSW.hpp"
#include "algo/algo.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"

using ANN::HNSW;

template<class U, class S1, class S2>
void run_post_hnsw(uint32_t dim, float m_l, uint32_t m, uint32_t efc, float alpha, float batch_base,
                   uint32_t k, uint32_t ef, size_t size_init, size_t size_step, size_t size_max,
                   const S1& ps, const S1& q, const S2& F_b, const S2& F_q) {
  HNSW<U> layers(dim, m_l, m, efc, alpha);

  puts("Initialize HNSW");
  parlay::internal::timer t;

  for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
       size_last = size_curr, size_curr += size_step) {
    printf("Increasing size from %lu to %lu\n", size_last, size_curr);

    puts("Insert points");
    auto ins_begin = ps.begin() + size_last;
    auto ins_end = ps.begin() + size_curr;

    layers.insert(ins_begin, ins_end, batch_base);
  }

  t.next("Finish insertion");

  puts("Collect statistics");
  print_layers(layers);

  puts("Search for neighbors and do post processing");
  auto res = post_processing(layers, q, k, ef, F_b, F_q);

  puts("Generate groundtruth");
  auto gt = ConstructKnng<U>(ps, q, dim, k, F_b, F_q);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}