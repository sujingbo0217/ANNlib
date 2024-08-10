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

template<class U, class Seq>
void run_hnsw(uint32_t dim, float m_l, uint32_t m, uint32_t efc, float alpha, float batch_base,
              uint32_t k, uint32_t ef, size_t size_init, size_t size_step, size_t size_max,
              const Seq& ps, const Seq& q) {
  // decltype(ps) baseset;
  HNSW<U> layers(dim, m_l, m, efc, alpha);
  // std::vector<vamana<U>> snapshots;
  puts("Initialize HNSW");
  parlay::internal::timer t;

  for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
       size_last = size_curr, size_curr += size_step) {
    printf("Increasing size from %lu to %lu\n", size_last, size_curr);

    puts("Insert points");
    auto ins_begin = ps.begin() + size_last;
    auto ins_end = ps.begin() + size_curr;

    layers.insert(ins_begin, ins_end, batch_base);
    // t.next("Finish insertion");

    // auto pids = std::ranges::subrange(ins_begin, ins_end) |
    //             std::views::take((size_curr - size_last) / 2) |
    //             std::views::transform([](const auto &p) { return p.get_id(); });
    // layers.erase(pids.begin(), pids.end());
    // t.next("Finish deletion");

    // snapshots.push_back(layers);
  }

  t.next("Finish insertion");

  puts("Collect statistics");
  print_layers(layers);

  puts("Search for neighbors");
  auto res = find_nbhs(layers, q, k, ef);

  puts("Generate groundtruth");
  auto gt = ConstructKnng<U>(ps, q, dim, k);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}