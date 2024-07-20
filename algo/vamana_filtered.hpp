#ifndef _ANN_ALGO_FILTERED_VAMANA_HPP
#define _ANN_ALGO_FILTERED_VAMANA_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include "algo/algo.hpp"
#include "custom/custom.hpp"
#include "map/direct.hpp"
#include "util/debug.hpp"
#include "util/helper.hpp"
#include "algo/vamana.hpp"
#include "algo/vamana_stitched.hpp"

namespace ANN {

  template<class Desc>
  class filtered_vamana : public stitched_vamana<Desc> {
  //  public:
  //   using stitched_vamana<Desc>::stitched_vamana;
  //   using cm = typename stitched_vamana<Desc>::cm;

  //   using nid_t = typename stitched_vamana<Desc>::nid_t;
  //   using pid_t = typename stitched_vamana<Desc>::pid_t;
  //   using coord_t = typename stitched_vamana<Desc>::coord_t;
  //   // using md_t = typename stitched_vamana<Desc>::md_t;
  //   using dist_t = typename stitched_vamana<Desc>::dist_t;
  //   using conn = typename stitched_vamana<Desc>::conn;
  //   using edge = typename stitched_vamana<Desc>::edge;
  //   using search_control = typename stitched_vamana<Desc>::search_control;
  //   using prune_control = typename stitched_vamana<Desc>::prune_control;

  //   using node_t = typename stitched_vamana<Desc>::node_t;
  //   using point_t = typename stitched_vamana<Desc>::point_t;
  //   using graph_t = typename stitched_vamana<Desc>::graph_t;
  //   using label_t = typename stitched_vamana<Desc>::label_t;
  //   using result_t = typename stitched_vamana<Desc>::result_t;

  //   template<typename T>
  //   using seq = typename cm::seq<T>;
  //   using seq_edge = seq<edge>;
  //   using seq_conn = seq<conn>;

   public:
    filtered_vamana(uint32_t dim, uint32_t R, uint32_t L, float alpha)
        : stitched_vamana<Desc>(dim, R, L, alpha) {}

  //  public:
  //   // template<typename Iter>
  //   // void insert(Iter begin, Iter end, float batch_base = 2);

  //   template<typename Iter>
  //   void insert(Iter begin, Iter end, const std::vector<std::vector<label_t>> &F,
  //               float batch_base = 2);

  //   void insert(const nid_t &nid, const coord_t &coord, const std::vector<label_t> &F);

  //   // template<class Seq = seq<result_t>>
  //   // Seq search(const coord_t &cq, uint32_t k, uint32_t ef, const search_control &ctrl = {})
  //   // const;

  //   template<class Seq = seq<result_t>>
  //   Seq search(const coord_t &cq, uint32_t k, uint32_t ef, const std::vector<label_t> &F,
  //              const search_control &ctrl = {}) const;

  //   //  public:
  //   //   static seq<edge> &&edge_cast(seq<conn> &&cs) {
  //   //     return reinterpret_cast<seq<edge> &&>(std::move(cs));
  //   //   }
  //   //   static const seq<edge> &edge_cast(const seq<conn> &cs) {
  //   //     return reinterpret_cast<const seq<edge> &>(cs);
  //   //   }
  //   //   static seq<conn> &&conn_cast(seq<edge> &&es) {
  //   //     return reinterpret_cast<seq<conn> &&>(std::move(es));
  //   //   }
  //   //   static const seq<conn> &conn_cast(const seq<edge> &es) {
  //   //     return reinterpret_cast<const seq<conn> &>(es);
  //   //   }

  //   //  private:
  //   //   struct node_t {
  //   //     coord_t coord;
  //   //     std::vector<label_t> labels;

  //   //     coord_t &get_coord() {
  //   //       return coord;
  //   //     }
  //   //     const coord_t &get_coord() const {
  //   //       return coord;
  //   //     }
  //   //     const std::vector<label_t> &get_label() const {
  //   //       return labels;
  //   //     }
  //   //   };

  //   //   using graph_t = typename Desc::graph_t<nid_t, node_t, edge>;

  //  public:
  //   graph_t g;
  //   seq<nid_t> entrance;  // To init
  //   std::unordered_set<nid_t> existed_points;

  //  private:
  //   map::direct<pid_t, nid_t> id_map;
  //   uint32_t dim;
  //   uint32_t R;
  //   uint32_t L;
  //   float alpha;

  //   // template<typename Iter>
  //   // void insert_batch_impl(Iter begin, Iter end);
  //   template<typename Iter>
  //   void insert_batch_impl(Iter begin, Iter end, const std::vector<std::vector<label_t>> &F);

  //  public:
  //   //   uint32_t get_deg_bound() const {
  //   //     return R;
  //   //   }

  //   //  public:
  //   //   auto gen_f_dist(const coord_t &c) const {
  //   //     class dist_evaluator {
  //   //       std::reference_wrapper<const graph_t> g;
  //   //       std::reference_wrapper<const coord_t> c;
  //   //       uint32_t dim;

  //   //      public:
  //   //       dist_evaluator(const graph_t &g, const coord_t &c, uint32_t dim) : g(g), c(c), dim(dim)
  //   //       {} dist_t operator()(nid_t v) const {
  //   //         return Desc::distance(c, g.get().get_node(v)->get_coord(), dim);
  //   //       }
  //   //       dist_t operator()(nid_t u, nid_t v) const {
  //   //         return Desc::distance(g.get().get_node(u)->get_coord(),
  //   //         g.get().get_node(v)->get_coord(),
  //   //                               dim);
  //   //       }
  //   //     };

  //   //     return dist_evaluator(g, c, dim);
  //   //   }
  //   //   auto gen_f_dist(nid_t u) const {
  //   //     return gen_f_dist(g.get_node(u)->get_coord());
  //   //   }

  //   //   template<class G>
  //   //   auto gen_f_nbhs(const G &g) const {
  //   //     return [&](nid_t u) -> decltype(auto) {
  //   //       // TODO: use std::views::transform in C++20 / util::tranformed_view
  //   //       // TODO: define the return type as a view, and use auto to receive it
  //   //       if constexpr (std::is_reference_v<decltype(g.get_edges(u))>) {
  //   //         const auto &edges = g.get_edges(u);
  //   //         return util::delayed_seq(edges.size(), [&](size_t i) { return edges[i].u; });
  //   //       } else {
  //   //         auto edges = g.get_edges(u);
  //   //         return util::delayed_seq(edges.size(), [=](size_t i) { return edges[i].u; });
  //   //       }
  //   //     };
  //   //   }

  //   template<class G>
  //   auto get_f_label(const G &g) const {
  //     return [&](nid_t u) -> decltype(auto) { return g.get_node(u)->get_label(); };
  //   }

  //   //   template<class Op>
  //   //   auto calc_degs(Op op) const {
  //   //     seq<size_t> degs(cm::num_workers(), 0);
  //   //     g.for_each([&](auto p) {
  //   //       auto &deg = degs[cm::worker_id()];
  //   //       deg = op(deg, g.get_edges(p).size());
  //   //     });
  //   //     return cm::reduce(degs, size_t(0), op);
  //   //   }

  //   //  public:
  //   //   size_t num_nodes() const {
  //   //     return g.num_nodes();
  //   //     // std::cerr << "total points: " << existed_points.size() << '\n';
  //   //     // return existed_points.size();
  //   //   }

  //   //   size_t num_edges(nid_t u) const {
  //   //     return g.get_edges(u).size();
  //   //   }

  //   //   size_t num_edges() const {
  //   //     return calc_degs(std::plus<>{});
  //   //   }

  //   //   size_t max_deg() const {
  //   //     return calc_degs([](size_t x, size_t y) { return std::max(x, y); });
  //   //   }

  //   bool is_point_existed(pid_t pid) const {
  //     return (existed_points.find(pid) != existed_points.end());
  //   }

  //   bool is_node_existed(nid_t nid) const {
  //     return is_node_existed(id_map.get_pid(nid));
  //   }

  //   bool set_point(pid_t pid) {
  //     if (!is_point_existed(pid)) {
  //       existed_points.insert(pid);
  //       return false;
  //     }
  //     return true;
  //   }

  //   bool set_node(nid_t nid) {
  //     return set_point(id_map.get_pid(nid));
  //   }
  };

  // template<class Desc>
  // template<typename Iter>
  // void filtered_vamana<Desc>::insert(Iter begin, Iter end,
  //                                    const std::vector<std::vector<label_t>> &F, float batch_base) {
  //   static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type,
  //                                typename stitched_vamana<Desc>::point_t>);
  //   static_assert(std::is_base_of_v<std::random_access_iterator_tag,
  //                                   typename std::iterator_traits<Iter>::iterator_category>);

  //   const size_t n = std::distance(begin, end);
  //   if (n == 0) return;

  //   // std::random_device rd;
  //   auto perm = cm::random_permutation(n /*, rd()*/);
  //   auto rand_seq =
  //       util::delayed_seq(n, [&](size_t i) -> decltype(auto) { return *(begin + perm[i]); });
  //   // auto rand_label_seq =
  //   //     util::delayed_seq(n, [&](size_t i) -> decltype(auto) { return *(F.begin() + perm[i]); });

  //   for (auto it = begin; it != end; it++) {
  //     // set_point(it->first);
  //     set_point(it->get_id());
  //   }

  //   // std::cerr << "total nodes now: " << existed_points.size() << '\n';

  //   size_t cnt_skip = 0;
  //   if (g.empty()) {
  //     // const nid_t ep_init = id_map.insert(rand_seq.begin()->get_id());
  //     auto init = rand_seq.begin();
  //     const nid_t ep = id_map.insert(static_cast<pid_t>(init->get_id()));
  //     g.add_node(ep, node_t{init->get_coord(), *(F.begin())});
  //     // const nid_t ep_init = id_map.insert(static_cast<pid_t>(it->get_id()));
  //     // g.add_node(ep_init, node_t{it->get_coord(), *(F.begin())});
  //     entrance.push_back(ep);
  //     // additional entry point
  //     // it++;
  //     // const nid_t ep_second = id_map.insert(static_cast<pid_t>(it->first));
  //     // g.add_node(ep_second, node_t{it->second.get_coord(), *(F.begin() + 1)});
  //     // entrance.push_back(ep_second);
  //     cnt_skip = 1;
  //   }

  //   size_t batch_begin = 0, batch_end = cnt_skip, size_limit = std::max<size_t>(n * 0.02, 20000);
  //   float progress = 0.0;
  //   while (batch_end < n) {
  //     batch_begin = batch_end;
  //     batch_end =
  //         std::min<size_t>({n, (size_t)std::ceil(batch_begin * batch_base) + 1, batch_begin + size_limit});

  //     // std::cerr << "(batch_begin, batch_end)" << batch_begin << " " << batch_end << '\n';

  //     util::debug_output("Batch insertion: [%u, %u)\n", batch_begin, batch_end);
  //     // insert_batch_impl(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end);
  //     insert_batch_impl(rand_seq.begin() + batch_begin, rand_seq.begin() + batch_end, F);
  //     // insert(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end, false);

  //     // if (batch_end > n * (progress + 0.05)) {
  //     //   progress = float(batch_end) / n;
  //     //   fprintf(stderr, "Built: %3.2f%%\n", progress * 100);
  //     //   fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
  //     //   fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
  //     //   fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
  //     //   per_visited.clear();
  //     //   per_eval.clear();
  //     //   per_size_C.clear();
  //     // }
  //   }

  //   // fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
  //   // fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
  //   // fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
  //   // per_visited.clear();
  //   // per_eval.clear();
  //   // per_size_C.clear();
  // }

  // template<class Desc>
  // void filtered_vamana<Desc>::insert(const nid_t &nid, const coord_t &coord,
  //                                    const std::vector<label_t> &F) {
  //   id_map.insert(static_cast<pid_t>(nid));
  //   g.add_node(nid, node_t{coord, F});
  // }

  // template<class Desc>
  // template<typename Iter>
  // void filtered_vamana<Desc>::insert_batch_impl(Iter begin, Iter end,
  //                                               const std::vector<std::vector<label_t>> &F) {
  //   const size_t size_batch = std::distance(begin, end);
  //   seq<nid_t> nids(size_batch);

  //   // per_visited.resize(size_batch);
  //   // per_eval.resize(size_batch);
  //   // per_size_C.resize(size_batch);

  //   // before the insertion, prepare the needed data
  //   // `nids[i]` is the nid of the node corresponding to the i-th
  //   // point to insert in the batch, associated with level[i]
  //   id_map.insert(util::delayed_seq(size_batch, [&](size_t i) {
  //     return (begin + i)->get_id();
  //   }));

  //   cm::parallel_for(0, size_batch, [&](uint32_t i) {
  //     nids[i] = id_map.get_nid((begin + i)->get_id());
  //   });

  //   g.add_nodes(util::delayed_seq(size_batch, [&](size_t i) {
  //     // GUARANTEE: begin[*].get_coord is only invoked for assignment once
  //     return std::pair{nids[i], node_t{(begin + i)->get_coord(), *(F.begin() + i)}};
  //     // return std::pair{nids[i], node_t{(begin + i)->get_coord(), *(F.begin() + i)}};
  //   }));

  //   // below we (re)generate edges incident to nodes in the current batch
  //   // add adges from the new points
  //   seq<seq<std::pair<nid_t, edge>>> edge_added(size_batch);
  //   seq<std::pair<nid_t, seq<edge>>> nbh_forward(size_batch);
  //   cm::parallel_for(0, size_batch, [&](size_t i) {
  //     const nid_t u = nids[i];

  //     auto &eps_u = entrance;
  //     search_control sctrl;  // TODO: use designated initializers in C++20
  //     sctrl.log_per_stat = i;
  //     seq<conn> res =
  //         algo::beamSearch(gen_f_nbhs(g), gen_f_dist(u), get_f_label(g), eps_u, L, F[i], sctrl);

  //     prune_control pctrl;  // TODO: use designated intializers in C++20
  //     pctrl.alpha = alpha;
  //     seq<conn> conn_u = algo::prune_heuristic(std::move(res), vamana<Desc>::get_deg_bound(), vamana<Desc>::gen_f_nbhs(g),
  //                                              vamana<Desc>::gen_f_dist(u), pctrl);
  //     // record the edge for the backward insertion later
  //     auto &edge_cur = edge_added[i];
  //     edge_cur.clear();
  //     edge_cur.reserve(conn_u.size());
  //     for (const auto &[d, v] : conn_u) {
  //       edge_cur.emplace_back(v, edge{d, u});
  //     }

  //     // store for batch insertion
  //     nbh_forward[i] = {u, edge_cast(std::move(conn_u))};
  //   });
  //   util::debug_output("Adding forward edges\n");
  //   g.set_edges(std::move(nbh_forward));

  //   // now we add edges in the other direction
  //   auto edge_added_flatten = util::flatten(std::move(edge_added));
  //   auto edge_added_grouped = util::group_by_key(std::move(edge_added_flatten));

  //   // TODO: use std::remove_cvref in C++20
  //   using agent_t = std::remove_cv_t<std::remove_reference_t<decltype(g.get_edges(nid_t()))>>;
  //   seq<std::pair<nid_t, agent_t>> nbh_backward(edge_added_grouped.size());

  //   cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j) {
  //     nid_t v = edge_added_grouped[j].first;
  //     auto &nbh_v_add = edge_added_grouped[j].second;

  //     auto edge_agent_v = g.get_edges(v);
  //     auto edge_v = util::to<seq<edge>>(std::move(edge_agent_v));
  //     edge_v.insert(edge_v.end(), std::make_move_iterator(nbh_v_add.begin()),
  //                   std::make_move_iterator(nbh_v_add.end()));

  //     seq<conn> conn_v = algo::prune_simple(conn_cast(std::move(edge_v)), vamana<Desc>::get_deg_bound());
  //     edge_agent_v = edge_cast(conn_v);
  //     nbh_backward[j] = {v, std::move(edge_agent_v)};
  //   });
  //   util::debug_output("Adding backward edges\n");
  //   g.set_edges(std::move(nbh_backward));

  //   // finally, update the entrances
  //   util::debug_output("Updating entrance\n");
  //   // UNIMPLEMENTED
  // }

  // template<class Desc>
  // template<class Seq>
  // Seq filtered_vamana<Desc>::search(const coord_t &cq, uint32_t k, uint32_t ef,
  //                                   const std::vector<label_t> &F,
  //                                   const search_control &ctrl) const {
  //   seq<nid_t> eps = entrance;
  //   // auto nbhs = beamSearch(gen_f_nbhs(g), gen_f_dist(cq), eps, ef, ctrl);
  //   auto nbhs = beamSearch(gen_f_nbhs(g), gen_f_dist(cq), get_f_label(g), eps, ef, F, ctrl);

  //   nbhs = algo::prune_simple(std::move(nbhs), k /*, ctrl*/);  // TODO: set ctrl
  //   cm::sort(nbhs.begin(), nbhs.end());

  //   using result_t = typename Seq::value_type;
  //   static_assert(util::is_direct_list_initializable_v<result_t, dist_t, pid_t>);
  //   Seq res(nbhs.size());
  //   cm::parallel_for(0, nbhs.size(), [&](size_t i) {
  //     const auto &nbh = nbhs[i];
  //     res[i] = result_t{nbh.d, id_map.get_pid(nbh.u)};
  //   });

  //   return res;
  // }

}  // namespace ANN

#endif  // _ANN_ALGO_FILTERED_VAMANA_HPP
