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
#include "algo/hnsw_filtered.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"

using ANN::filtered_hnsw;

template<typename U, typename L>
void run_filtered_hnsw(uint32_t dim, float m_l, uint32_t m, uint32_t efc, float alpha,
                       float batch_base, uint32_t k, uint32_t ef, size_t size_init,
                       size_t size_step, size_t size_max, auto ps, auto q, auto F_b, auto P_b,
                       auto F_q, /*auto P_q,*/ bool specificity = false) {
  using nid_t = typename filtered_hnsw<U>::nid_t;
  using pid_t = typename filtered_hnsw<U>::pid_t;

  filtered_hnsw<U> layers(dim, m_l, m, efc, alpha);
  // std::vector<filtered_hnsw<U>> snapshots;
  puts("Initialize Filtered HNSW");

  auto FindMedoid = [&]() -> std::vector<nid_t> {
    // parallelism
    size_t n = q.size();
    std::vector<nid_t> M(n);

    parlay::parallel_for(0, n, [&](size_t i) {
      const std::vector<L> &F = F_q[i];
      size_t m = F.size();
      auto f_dist = layers.gen_f_dist(q[i].get_coord());
      std::vector<pid_t> temp(m);

      parlay::parallel_for(0, m, [&](size_t j) {
        L f = F[j];
        pid_t minv = n;
        if (P_b.find(f) != P_b.end()) {
          const std::vector<pid_t> &p = P_b[f];
          for (const pid_t &v : p) {
            const auto d = f_dist(v);
            if (minv == n || d < f_dist(minv)) {
              minv = v;
            }
          }
        }
        temp[j] = minv;
      });

      pid_t minv = n;
      for (const pid_t &v : temp) {
        const auto d = f_dist(v);
        if (minv == n || d < f_dist(minv)) {
          minv = v;
        }
      }

      M[i] = layers.id_map.get_nid(minv);
    });

    return M;
  };

  parlay::internal::timer t;

  for (size_t size_last = 0, size_curr = size_init; size_curr <= size_max;
       size_last = size_curr, size_curr += size_step) {
    printf("Increasing size from %lu to %lu\n", size_last, size_curr);

    puts("Insert points");
    parlay::internal::timer t("run_test:insert", true);

    auto ins_begin = ps.begin() + size_last;
    auto ins_end = ps.begin() + size_curr;

    auto insert_labels = ANN::util::to<decltype(F_b)>(
        std::ranges::subrange(F_b.begin() + size_last, F_b.begin() + size_curr));

    layers.insert(ins_begin, ins_end, insert_labels, batch_base);

    // snapshots.push_back(layers);

    puts("Collect statistics");
    print_layers(layers);
  }

  t.next("Finish insertion");

  puts("Search for neighbors");
  layers.entrance.clear();
  auto medoid = FindMedoid();
  layers.entrance =
      ANN::util::to<decltype(layers.entrance)>(std::ranges::subrange(medoid.begin(), medoid.end()));
  auto res = find_nbhs(layers, q, k, ef, F_q, true);

  puts("Generate groundtruth");
  // auto baseset = ANN::util::to<decltype(ps)>(std::ranges::subrange(ps.begin(), ins_end));
  // auto base_labels = ANN::util::to<decltype(F_b)>(std::ranges::subrange(F_b.begin(), lb_end));
  auto gt = ConstructKnng<U>(ps, q, dim, k, F_b, F_q);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------");
}