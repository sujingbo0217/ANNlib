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

template<class U, class S1, class S2, class S3>
auto run_stitched_vamana_insert(uint32_t dim, uint32_t m, uint32_t efc, float alpha,
                                float batch_base, size_t size_max, const S1 &ps, const S2 &F_b,
                                const S3 &P_b) {
  using nid_t = typename stitched_vamana<U>::nid_t;
  using pid_t = typename stitched_vamana<U>::pid_t;
  using label_t = typename stitched_vamana<U>::label_t;
  using seq_edge = typename stitched_vamana<U>::seq_edge;
  using seq_conn = typename stitched_vamana<U>::seq_conn;
  using prune_control = typename stitched_vamana<U>::prune_control;
  using cm = typename stitched_vamana<U>::cm;
  // constexpr bool filtered = false;

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

  std::vector<std::pair<label_t, std::vector<nid_t>>> Pb(P_b.begin(), P_b.end());
  sort(Pb.begin(), Pb.end(),
       [&](const std::pair<label_t, std::vector<nid_t>> &a,
           const std::pair<label_t, std::vector<nid_t>> &b) {
         return a.second.size() < b.second.size();
       });
  const size_t n_unique_label = Pb.size();
  stitched_vamana<U> base(dim, m + 32, efc + 50, alpha);  // R = R_small + 32, L = L_small + 50

  puts("Initialize Stitched Vamana");
  size_t idx = 0;

  // puts("Insert points");
  parlay::internal::timer t;

  for (const auto &[f, Pf] : Pb) {
    // if (Pf.size() == 1 && Pf[0] == 0) continue;
    size_t n = Pf.size();
    printf("Number of points w/ label [%u]: %lu\n", f, n);
    S1 new_ps(n);
    std::vector<std::vector<label_t>> base_labels(n);

    parlay::parallel_for(0, n, [&](size_t i) {
      auto offset = Pf[i];
      new_ps[i] = *(ps.begin() + offset);
      base_labels[i] = F_b[offset];
    });

    auto ins_begin = new_ps.begin();
    auto ins_end = new_ps.end();

    stitched_vamana<U> g(dim, m, efc, alpha);
    g.insert(ins_begin, ins_end, base_labels, batch_base /*, filtered*/);

    Merge(g, base);
    printf("Inserted points w/ label: %lu/%lu\n", ++idx, n_unique_label);
  }

  puts("Collect statistics");
  print_stat(base);

  t.next("Finish construction");

  return base;
}

template<class G, class E, class S1, class S2, class S3, class GT>
void run_stitched_vamana_search(G g, const E &medoid, uint32_t k, uint32_t ef, const S1 &q,
                                const S2 &F_q, const S3 &P_b, const GT &gt) {
  // constexpr bool filtered = false;
  parlay::internal::timer t;

  puts("Search for neighbors");
  g.entrance.clear();
  // auto medoid = g.template find_medoid(P_b, 0.2);
  g.entrance =
      ANN::util::to<decltype(g.entrance)>(std::ranges::subrange(medoid.begin(), medoid.end()));
  auto res = find_nbhs(g, q, k, ef, F_q /*, filtered*/);
  t.next("Finish searching");

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("--------------------------------\n");
}