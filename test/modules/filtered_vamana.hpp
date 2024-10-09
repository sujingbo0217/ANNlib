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
#include "algo/vamana_filtered.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"

using ANN::filtered_vamana;

template<class U, class S1, class S2, class S3>
auto run_filtered_vamana_insert(uint32_t dim, uint32_t m, uint32_t efc, float alpha,
                                float batch_base, size_t size_init, size_t size_step,
                                size_t size_max, const S1 &ps, const S2 &F_b, const S3 &medoid) {
  // using nid_t = typename filtered_vamana<U>::nid_t;
  // using pid_t = typename filtered_vamana<U>::pid_t;
  // using label_t = typename filtered_vamana<U>::label_t;
  // constexpr bool filtered = true;

  puts("Initialize Filtered Vamana");
  filtered_vamana<U> g(dim, m, efc, alpha);

  parlay::internal::timer t;

  for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
       size_last = size_curr, size_curr += size_step) {
    auto ins_begin = ps.begin() + size_last;
    auto ins_end = ps.begin() + size_curr;

    const auto &insert_labels =
        ANN::util::to<S2>(std::ranges::subrange(F_b.begin() + size_last, F_b.begin() + size_curr));

    g.entrance.clear();
    g.entrance =
        ANN::util::to<decltype(g.entrance)>(std::ranges::subrange(medoid.begin(), medoid.end()));
    g.insert(ins_begin, ins_end, insert_labels, batch_base /*, filtered*/);
    std::cout << "Insert from " << size_last << " to " << size_curr << std::endl;
  }

  puts("Collect statistics");
  print_stat(g);

  t.next("Finish insertion");

  return g;
}

template<class G, class E, class S1, class S2, class S3, class GT>
void run_filtered_vamana_search(G g, const E &medoid, uint32_t k, uint32_t ef, const S1 &q,
                                const S2 &F_q, const S3 &P_b, const GT &gt) {
  // constexpr bool filtered = true;
  parlay::internal::timer t;

  puts("Search for neighbors");
  // g.entrance.clear();
  // auto medoid = g.template find_medoid(P_b, 0.2);
  // g.entrance =
  //     ANN::util::to<decltype(g.entrance)>(std::ranges::subrange(medoid.begin(), medoid.end()));
  auto res = find_nbhs(g, q, k, ef, F_q /*, filtered*/);
  t.next("Finish searching");

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}