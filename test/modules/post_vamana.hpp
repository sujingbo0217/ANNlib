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

template<typename U, typename L>
void run_post_vamana(uint32_t dim, uint32_t m, uint32_t efc, float alpha, float batch_base,
                     uint32_t k, uint32_t ef, size_t size_init, size_t size_step, size_t size_max,
                     auto ps, auto q, auto F_b, auto F_q) {
  vamana<U> g(dim, m, efc, alpha);

  puts("Initialize Vamana");

  for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
       size_last = size_curr, size_curr += size_step) {
    printf("Increasing size from %lu to %lu\n", size_last, size_curr);

    puts("Insert points");
    parlay::internal::timer t("run_test:insert", true);

    auto ins_begin = ps.begin() + size_last;
    auto ins_end = ps.begin() + size_curr;

    g.insert(ins_begin, ins_end, batch_base);
    t.next("Finish insertion");
  }

  puts("Collect statistics");
  print_stat(g);

  puts("Search for neighbors and do post processing");
  auto res = post_processing<U>(g, q, k, ef, F_b, F_q);

  puts("Generate groundtruth");

  // auto baseset = ANN::util::to<decltype(ps)>(std::ranges::subrange(ps.begin(), ins_end));
  auto gt = ConstructKnng<U>(ps, q, dim, k, F_b, F_q);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------");
}