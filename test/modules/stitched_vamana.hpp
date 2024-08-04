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

#include "../parlay.hpp"
#include "../utils.hpp"
#include "algo/algo.hpp"
#include "algo/vamana_stitched.hpp"
#include "graph/adj.hpp"
#include "util/intrin.hpp"

using ANN::stitched_vamana;

template<typename U, typename L>
void run_stitched_vamana(uint32_t dim, uint32_t m, uint32_t efc, float alpha, float batch_base,
                         uint32_t k, uint32_t ef, size_t size_init, size_t size_step,
                         size_t size_max, auto ps, auto q, auto F_b, auto P_b, auto F_q,
                         /*auto P_q,*/ bool specificity = false) {
  using nid_t = typename stitched_vamana<U>::nid_t;
  using pid_t = typename stitched_vamana<U>::pid_t;
  using seq_edge = typename stitched_vamana<U>::seq_edge;
  using seq_conn = typename stitched_vamana<U>::seq_conn;
  using prune_control = typename stitched_vamana<U>::prune_control;
  using cm = typename stitched_vamana<U>::cm;

  auto Merge = [alpha, size_max](stitched_vamana<U> from, stitched_vamana<U> &to) -> void {
    // entrance point set update
    for (const auto &ep : from.entrance) {
      to.entrance.push_back(ep);
    }
    std::sort(to.entrance.begin(), to.entrance.end());
    to.entrance.erase(std::unique(to.entrance.begin(), to.entrance.end()), to.entrance.end());

    const auto mapping = from.id_map.get_map();

    std::vector<std::pair<nid_t, pid_t>> existed_nodes, missing_nodes;

    for (const auto &node : mapping) {
      if (!to.id_map.has_node(node.first)) {
        missing_nodes.push_back(node);
      } else {
        existed_nodes.push_back(node);
      }
    }

    size_t n = missing_nodes.size();
    size_t m = existed_nodes.size();

    printf("Missing node num: %lu, Existed node num: %lu\n", n, m);

    typename stitched_vamana<U>::seq<std::pair<nid_t, seq_edge>> nbh_missing(n), nbh_existed(m);

    for (const auto &[nid, pid] : missing_nodes) {
      assert(pid < size_max);
      const auto &node = from.g.get_node(nid);
      const auto &labels = node->get_label();
      const auto &coord = node->get_coord();
      to.insert(std::make_pair(pid, nid), coord, labels);
    }

    // parallelism
    cm::parallel_for(0, n, [&](size_t i) {
      auto [nid, pid] = missing_nodes[i];
      assert(pid < size_max && to.id_map.has_node(nid));
      auto edges = from.g.get_edges(nid);
      // no need to prune
      nbh_missing[i] = std::make_pair(nid, edges);
    });
    to.g.set_edges(std::move(nbh_missing));

    cm::parallel_for(0, m, [&](size_t i) {
      auto [nid, pid] = existed_nodes[i];
      assert(pid < size_max);
      auto to_edges = to.g.get_edges(nid);
      const auto &from_edges = from.g.get_edges(nid);  // keep immutable here
      auto edge_v = ANN::util::to<seq_edge>(std::move(to_edges));
      edge_v.insert(edge_v.end(), std::make_move_iterator(from_edges.begin()),
                    std::make_move_iterator(from_edges.end()));
      prune_control pctrl;
      pctrl.alpha = alpha;
      seq_conn conn_v =
          // ANN::algo::prune_simple(to.conn_cast(std::move(edge_v)), to.get_deg_bound());
          ANN::algo::prune_heuristic(to.conn_cast(std::move(edge_v)), to.get_deg_bound(),
                                     to.gen_f_nbhs(), to.gen_f_dist(nid), pctrl);
      nbh_existed[i] = std::make_pair(nid, to.edge_cast(std::move(conn_v)));
    });
    to.g.set_edges(std::move(nbh_existed));
  };

  const size_t n_unique_label = P_b.size();
  stitched_vamana<U> base(dim, m * 2, efc * 2, alpha);
  constexpr bool filtered = false;
  // std::vector<stitched_vamana<U>> snapshots;

  puts("Initialize Stitched Vamana");
  size_t idx = 0;

  puts("Insert points");
  parlay::internal::timer t;

  for (const auto &[f, Pf] : P_b) {
    // if (Pf.size() == 1 && Pf[0] == 0) continue;
    size_t n = Pf.size();
    printf("Number of points w/ label [%u]: %lu\n", f, n);
    decltype(ps) new_ps(n);
    std::vector<std::vector<L>> base_labels(n);

    parlay::parallel_for(0, n, [&](size_t i) {
      auto offset = Pf[i];
      new_ps[i] = *(ps.begin() + offset);
      base_labels[i] = F_b[offset];
    });

    auto ins_begin = new_ps.begin();
    auto ins_end = new_ps.end();

    stitched_vamana<U> g(dim, m, efc, alpha);
    g.insert(ins_begin, ins_end, base_labels, batch_base, filtered);

    Merge(g, base);
    printf("Inserted points w/ label: %lu/%lu\n", ++idx, n_unique_label);

    puts("Collect statistics");
    print_stat(base);
  }

  t.next("Finish construction");

  std::vector<L> qLs = stitched_vamana<U>::get_specificity(P_b);

  if (specificity) {
    for (size_t i = 0; i < 5; ++i) {
      size_t j = 4 - i;
      printf("\n### Specificity: %ld pc.\n", (j == 0 ? 1 : j * 25));

      L qL = qLs[j];
      size_t n = P_b[qL].size();
      printf("Label [%u]: |Pf|/|P| = %.3f\n", qL, n / (float)size_max);

      std::vector<std::vector<L>> query_labels(n, std::vector<L>(1, qL));

      puts("Search for neighbors");
      auto res = find_nbhs(base, q, k, ef, query_labels, filtered);
      t.next("Finish searching");

      puts("Generate groundtruth");
      auto gt = ConstructKnng<U>(ps, q, dim, k, F_b, query_labels);

      puts("Compute recall");
      calc_recall(q, res, gt, k);
    }
    puts("--------------------------------");
  } else {
    puts("Search for neighbors");
    auto res = find_nbhs(base, q, k, ef, F_q, filtered);
    t.next("Finish searching");

    puts("Generate groundtruth");
    auto gt = ConstructKnng<U>(ps, q, dim, k, F_b, F_q);

    puts("Compute recall");
    calc_recall(q, res, gt, k);

    puts("--------------------------------");
  }
}