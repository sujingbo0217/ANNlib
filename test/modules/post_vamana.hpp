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
#include "algo/algo.hpp"
#include "algo/vamana.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"

using ANN::vamana;

template<class U, class S1, class S2>
void run_post_vamana(uint32_t dim, uint32_t m, uint32_t efc, float alpha, float batch_base,
                     uint32_t k, uint32_t ef, size_t size_init, size_t size_step, size_t size_max,
                     const S1 &ps, const S1 &q, const S2 &F_b, const S2 &F_q) {
  vamana<U> g(dim, m, efc, alpha);

  puts("Initialize Vamana");
  parlay::internal::timer t;

  for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
       size_last = size_curr, size_curr += size_step) {
    printf("Increasing size from %lu to %lu\n", size_last, size_curr);

    puts("Insert points");
    auto ins_begin = ps.begin() + size_last;
    auto ins_end = ps.begin() + size_curr;

    g.insert(ins_begin, ins_end, batch_base);
  }

  t.next("Finish insertion");

  puts("Collect statistics");
  print_stat(g);

  puts("Search for neighbors and do post processing");
  auto res = post_processing(g, q, k, ef, F_b, F_q);
  // auto res = find_nbhs(g, q, k, ef);

  puts("Generate groundtruth");

  auto gt = ConstructKnng<U>(ps, q, dim, k, F_b, F_q);
  // auto gt = ConstructKnng<U>(ps, q, dim, k);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}