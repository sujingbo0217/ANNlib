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

#include "algo/algo.hpp"
#include "algo/vamana.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"
#include "../utils.hpp"

using ANN::vamana;

template<class U, class Seq>
void run_vamana(uint32_t dim, uint32_t m, uint32_t efc, float alpha, float batch_base, uint32_t k,
            uint32_t ef, size_t size_init, size_t size_step, size_t size_max, const Seq &ps, const Seq &q) {
  // decltype(ps) baseset;
  vamana<U> g(dim, m, efc, alpha);
  // std::vector<vamana<U>> snapshots;
  puts("Initialize Vamana");

  parlay::internal::timer t;

  for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
       size_last = size_curr, size_curr += size_step) {
    printf("Increasing size from %lu to %lu\n", size_last, size_curr);

    puts("Insert points");
    auto ins_begin = ps.begin() + size_last;
    auto ins_end = ps.begin() + size_curr;

    g.insert(ins_begin, ins_end, batch_base);
    // t.next("Finish insertion");

    // auto pids = std::ranges::subrange(ins_begin, ins_end) |
    //             std::views::take((size_curr - size_last) / 2) |
    //             std::views::transform([](const auto &p) { return p.get_id(); });
    // g.erase(pids.begin(), pids.end());
    // t.next("Finish deletion");

    // snapshots.push_back(g);
  }

  t.next("Finish insertion");

  puts("Collect statistics");
  print_stat(g);

  puts("Search for neighbors");
  auto res = find_nbhs(g, q, k, ef);

  puts("Generate groundtruth");

  // baseset.append(ins_begin + (size_curr - size_last) / 2, ins_end);
  // auto baseset = ANN::util::to<decltype(ps)>(std::ranges::subrange(ps.begin(), ins_end));
  auto gt = ConstructKnng<U>(ps, q, dim, k);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}