#ifndef _ANN_ALGO_STITCHED_VAMANA_HPP
#define _ANN_ALGO_STITCHED_VAMANA_HPP

#include <utility>
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
#include <vector>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "algo/algo.hpp"
#include "algo/vamana.hpp"
#include "custom/custom.hpp"
#include "map/direct.hpp"
#include "util/debug.hpp"
#include "util/helper.hpp"

namespace ANN::vamana_details {

template<typename Nid>
struct stitched_edge : util::conn<Nid> {
  constexpr bool operator<(const stitched_edge &rhs) const {
    return this->u < rhs.u;
  }
  constexpr bool operator>(const stitched_edge &rhs) const {
    return this->u > rhs.u;
  }
  constexpr bool operator==(const stitched_edge &rhs) const {
    return this->u == rhs.u;
  }
  constexpr bool operator!=(const stitched_edge &rhs) const {
    return this->u != rhs.u;
  }
};

}  // namespace ANN::vamana_details

template<typename Nid>
struct std::hash<ANN::vamana_details::stitched_edge<Nid>> {
  size_t operator()(const ANN::vamana_details::stitched_edge<Nid> &e) const noexcept {
    return std::hash<decltype(e.u)>{}(e.u);
  }
};

namespace ANN {

template<class Desc>
class stitched_vamana : public vamana<Desc> {
 public:
  using vamana<Desc>::vamana;
  using cm = typename vamana<Desc>::cm;

  using nid_t = typename vamana<Desc>::nid_t;
  using pid_t = typename vamana<Desc>::pid_t;
  using coord_t = typename vamana<Desc>::coord_t;
  using point_t = typename vamana<Desc>::point_t;
  using label_t = typename point_t::label_t;

  // using md_t = typename vamana<Desc>::md_t;
  using dist_t = typename vamana<Desc>::dist_t;
  using conn = typename vamana<Desc>::conn;
  using edge = vamana_details::stitched_edge<nid_t>;
  using search_control = typename vamana<Desc>::search_control;
  using prune_control = typename vamana<Desc>::prune_control;
  using result_t = typename vamana<Desc>::result_t;

 private:
  struct node_t {
    coord_t coord;
    std::vector<label_t> labels;

    coord_t &get_coord() {
      return coord;
    }
    const coord_t &get_coord() const {
      return coord;
    }
    const std::vector<label_t> &get_label() const {
      return labels;
    }
  };

  using graph_t = typename Desc::graph_aux<nid_t, node_t, edge>;

 public:
  template<typename T>
  using seq = typename cm::seq<T>;
  using seq_edge = seq<edge>;
  using seq_conn = seq<conn>;

 public:
  stitched_vamana(uint32_t dim, uint32_t R, uint32_t L, float alpha)
      : vamana<Desc>(dim, R, L, alpha), dim(dim), R(R), L(L), alpha(alpha) {}

 public:
  template<typename Iter, class Label>
  void insert(Iter begin, Iter end, const Label &F,
              float batch_base = 2/*, bool filtered = false*/);

  void insert(const std::pair<pid_t, nid_t> &nid, const coord_t &coord,
              const std::vector<label_t> &F);

  // template<typename Iter, class E>
  // void insert(Iter begin, Iter end, const std::vector<std::vector<label_t>> &F,
  //             const E& medoid, float batch_base = 2/*, bool filtered = false*/);

  template<typename Iter, class Label, class E>
  void insert2(Iter begin, Iter end, const Label &F, const E &medoid, float batch_base = 2);

  template<class Seq = seq<result_t>>
  Seq search(const coord_t &cq, uint32_t k, uint32_t ef, const std::vector<label_t> &F,
             const search_control &ctrl = {}) const;

 public:
  static seq_edge &&edge_cast(seq_conn &&cs) {
    return reinterpret_cast<seq_edge &&>(std::move(cs));
  }
  static const seq_edge &edge_cast(const seq_conn &cs) {
    return reinterpret_cast<const seq_edge &>(cs);
  }
  static seq_conn &&conn_cast(seq_edge &&es) {
    return reinterpret_cast<seq_conn &&>(std::move(es));
  }
  static const seq_conn &conn_cast(const seq_edge &es) {
    return reinterpret_cast<const seq_conn &>(es);
  }

  template<typename T>
  static std::vector<label_t> get_specificity(const T &P) {
    size_t n = P.size();
    std::vector<label_t> ret;

    if constexpr (std::is_same_v<T, std::unordered_map<typename T::key_type, std::vector<pid_t>>>) {
      size_t i = 1;
      for (const auto &[f, _] : P) {
        if (i == n || i == (size_t)(n * 0.75) || i == (size_t)(n * 0.5) ||
            i == (size_t)(n * 0.25) || i == std::max<size_t>(1, (size_t)(n * 0.01))) {
          ret.push_back(f);
        }
        ++i;
      }
    } else if constexpr (std::is_same_v<T, std::vector<std::pair<typename T::value_type::first_type,
                                                                 std::vector<pid_t>>>>) {
      for (size_t i = 0; i <= 100; i += 25) {
        size_t j = (i == 0 ? std::max<size_t>(1, (size_t)(n * 0.01)) : (size_t)(i / 100 * n));
        ret.push_back(P[j].first);
      }
    }

    assert(ret.size() == 5);
    return ret;
  }

 public:
  graph_t g;
  std::unordered_map<label_t,nid_t> entrance;  // To init
  map::direct<pid_t, nid_t> id_map;

 private:
  uint32_t dim;
  uint32_t R;
  uint32_t L;
  float alpha;

  // template<typename Iter>
  // void insert_batch_impl(Iter begin, Iter end);
  template<typename Iter, class Label>
  void insert_batch_impl(Iter begin, Iter end, const Label &F/*, bool filtered*/);

  template<typename Iter, class Label, class E>
  void insert_batch_impl2(Iter begin, Iter end, const Label &F, const E &medoid);

 public:
  uint32_t get_deg_bound() const {
    return R;
  }

  auto gen_f_dist(const coord_t &c) const {
    class dist_evaluator {
      std::reference_wrapper<const graph_t> g;
      std::reference_wrapper<const coord_t> c;
      uint32_t dim;

     public:
      dist_evaluator(const graph_t &g, const coord_t &c, uint32_t dim) : g(g), c(c), dim(dim) {}
      dist_t operator()(nid_t v) const {
        return Desc::distance(c.get(), g.get().get_node(v)->get_coord(), dim);
      }
      dist_t operator()(nid_t u, nid_t v) const {
        return Desc::distance(g.get().get_node(u)->get_coord(), g.get().get_node(v)->get_coord(),
                              dim);
      }
    };

    return dist_evaluator(g, c, dim);
  }

  auto gen_f_dist(nid_t u) const {
    return gen_f_dist(g.get_node(u)->get_coord());
  }

  auto gen_f_nbhs() const {
    return [&](nid_t u) {
      // auto f = std::views::filter([&](const edge &e) {
      //   auto &ls = e.livestamp;
      //   if (ls == 0) return false;
      //   if (ls == deltick) return true;
      //   ls = id_map.contain_nid(e.u) ? deltick : 0;
      //   return ls != 0;
      // });

      auto t = std::views::transform([&](const edge &e) {
        return e.u;
      });

      if constexpr (std::is_reference_v<decltype(g.get_edges(u))>) {
        return std::ranges::ref_view(g.get_edges(u)) /*| f */ | t;
      } else {
        return std::ranges::owning_view(g.get_edges(u)) /*| f */ | t;
      }
    };
  }

  // template<class G>
  auto get_f_label(/*const G &g*/) const {
    return [&](nid_t u) -> decltype(auto) {
      return g.get_node(u)->get_label();
    };
  }

  template<class Op>
  auto calc_degs(Op op) const {
    seq<size_t> degs(cm::num_workers(), 0);
    g.for_each([&](auto p) {
      auto &deg = degs[cm::worker_id()];
      deg = op(deg, num_edges(p.get_id()));
    });
    return cm::reduce(degs, size_t(0), op);
  }

  size_t num_nodes() const {
    return g.num_nodes();
  }

  size_t num_edges(nid_t u) const {
    // return g.get_edges(u).size();
    return std::ranges::distance(gen_f_nbhs()(u));
  }
  size_t num_edges() const {
    return calc_degs(std::plus<>{});
  }

  size_t max_deg() const {
    return calc_degs([](size_t x, size_t y) {
      return std::max(x, y);
    });
  }
};

template<class Desc>
template<typename Iter, class Label>
void stitched_vamana<Desc>::insert(Iter begin, Iter end, const Label &F, float batch_base/*, bool filtered*/) {

  static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type, point_t>);
  static_assert(std::is_base_of_v<std::random_access_iterator_tag,
                                  typename std::iterator_traits<Iter>::iterator_category>);
  using namespace std::views;

  const size_t n = std::distance(begin, end);
  if (n == 0) return;
  assert(F.size() == n);

  // std::random_device rd;
  // auto perm = cm::random_permutation(n /*, rd()*/);
  auto perm = util::to<seq<size_t>>(
    util::delayed_seq(n, [](size_t i){return i;})
  );

  auto f_full = cm::flatten(util::delayed_seq(F.size(), [&](size_t i){
    return F[perm[i]] | transform([i](label_t l){
      return std::pair(l, i);
    });
  }));
  cm::sort(f_full.begin(), f_full.end());
  auto f_missing = cm::pack(
    f_full,
    util::delayed_seq(f_full.size(), [&](size_t i){
      label_t l = f_full[i].first;
      return (i==0 || f_full[i-1].first!=l) && !entrance.contains(l);
  }));

  auto idx_cand = util::to<seq<size_t>>(
    f_missing | transform([](const auto &f){return f.second;})
  );
  cm::sort(idx_cand.begin(), idx_cand.end());
  auto idx_prio = cm::pack(
    idx_cand,
    util::delayed_seq(idx_cand.size(), [&](size_t i){
      return i==0 || idx_cand[i-1]!=idx_cand[i];
  }));

  // TODO: validate perm
  // auto rand_seq = util::delayed_seq(n, [&](size_t i) -> decltype(auto) {
  //   return *(begin + perm[i]);
  // });
/*
  size_t cnt_skip = 0;
  if (g.empty()) {
    // const nid_t ep_init = id_map.insert(rand_seq.begin()->get_id());
    auto init = rand_seq.begin();
    const nid_t ep = id_map.insert(init->get_id());
    g.add_node(ep, node_t{init->get_coord(), *(F.begin())});
    // const nid_t ep_init = id_map.insert(static_cast<pid_t>(it->get_id()));
    // g.add_node(ep_init, node_t{it->get_coord(), *(F.begin())});
    entrance.push_back(ep);
    cnt_skip = 1;
  }
*/
  size_t cnt_skip = idx_prio.size();

  id_map.insert(util::delayed_seq(
    cnt_skip, [&](size_t i){return (begin+perm[idx_prio[i]])->get_id();}
  ));
  auto eps_missing = f_missing | transform([&](const auto &f){
    const pid_t &pid = (begin+perm[f.second])->get_id();
    nid_t nid = id_map.get_nid(pid);
    return std::pair(f.first, nid);
  });
  entrance.insert(eps_missing.begin(), eps_missing.end());
/*{
  auto offset = std::lower_bound(idx_prio.begin(), idx_prio.end(), cnt_skip) - idx_prio.begin();
  cm::parallel_for(0, offset, [&](size_t i){
    perm[i] = perm[idx_prio[i]];
  });
  auto tmp = perm;
  cm::parallel_for(offset, cnt_skip, [&](size_t i){
    perm[i] = tmp[idx_prio[i]];
    perm[idx_prio[i]] = perm[i];
  });
}*/
  {
/*
    auto tmp = perm;
    cm::parallel_for(0, idx_prio.size(), [&](size_t i){
    // for(size_t i=0; i<idx_prio.size(); ++i){
      tmp[i] = perm[idx_prio[i]];
      tmp[idx_prio[i]] = perm[i];
      // std::swap(perm[i], perm[idx_prio[i]]);
    });
    // perm = tmp;
*/
    for(size_t i=0; i<idx_prio.size(); ++i){
      std::swap(perm[i], perm[idx_prio[i]]);
    }
/*
    for(size_t i=0; i<idx_prio.size(); ++i){
      assert(tmp[i] == tmp2[i]);
    }
*/
  }

  auto nids = util::to<seq<nid_t>>(util::delayed_seq(cnt_skip, [&](size_t i){
    return id_map.get_nid((begin+perm[i])->get_id());
  }));

  g.add_nodes(util::delayed_seq(cnt_skip, [&](size_t i){
    // nid_t nid = id_map.get_nid((begin+perm[i])->get_id());
    // GUARANTEE: begin[*].get_coord is only invoked for assignment once
    return std::pair{nids[i], node_t{(begin+perm[i])->get_coord(), F[perm[i]]}};
    // return std::pair{nids[i], node_t{(begin + i)->get_coord(), *(F.begin() + i)}};
  }));

  seq<std::pair<nid_t, seq_edge>> nbh_eps(cnt_skip);
  cm::parallel_for(0, cnt_skip, [&](size_t i){
    seq_conn res = util::to<seq_conn>(util::delayed_seq(cnt_skip, [&](size_t j){
      nid_t u=nids[i], v=nids[j];
      const auto &cu = g.get_node(u)->get_coord();
      const auto &cv = g.get_node(v)->get_coord();
      // NOTE: self circle
      return conn{Desc::distance(cu,cv,dim), v};
    }));
    seq_conn conn_u = algo::prune_simple(std::move(res), get_deg_bound());
    nbh_eps[i] = {nids[i], edge_cast(std::move(conn_u))};
  });
  g.set_edges(std::move(nbh_eps));

  size_t batch_begin = 0, batch_end = cnt_skip;
  size_t batch_step = 0, size_limit = std::max<size_t>(n * 0.02, 20000);
  // float progress = 0.0;
  auto rand_seq = util::delayed_seq(n, [&](size_t i) -> decltype(auto) {
    return *(begin + perm[i]);
  });

  while (batch_end < n) {
    batch_begin = batch_end;
    batch_step = std::min((size_t)std::ceil(batch_step*batch_base+1), size_limit);
    batch_end = std::min<size_t>(n, batch_begin+batch_step);

    // std::cerr << "(batch_begin, batch_end)" << batch_begin << " " << batch_end << '\n';

    util::debug_output("Batch insertion: [%u, %u)\n", batch_begin, batch_end);
    // insert_batch_impl(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end);
    auto new_labels = util::delayed_seq(batch_end-batch_begin, [&](size_t i){
      return F[perm[i+batch_begin]];
    });
    // insert_batch_impl(rand_seq.begin() + batch_begin, rand_seq.begin() + batch_end, new_labels/*, filtered*/);
    insert_batch_impl(rand_seq.begin() + batch_begin, rand_seq.begin() + batch_end, new_labels/*, filtered*/);
    // insert(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end, false);

    // if (batch_end > n * (progress + 0.05)) {
    //   progress = float(batch_end) / n;
    //   fprintf(stderr, "Built: %3.2f%%\n", progress * 100);
    //   fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
    //   fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
    //   fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
    //   per_visited.clear();
    //   per_eval.clear();
    //   per_size_C.clear();
    // }
  }

  // fprintf(stderr, "# visited: %lu\n", cm::reduce(per_visited));
  // fprintf(stderr, "# eval: %lu\n", cm::reduce(per_eval));
  // fprintf(stderr, "size of C: %lu\n", cm::reduce(per_size_C));
  // per_visited.clear();
  // per_eval.clear();
  // per_size_C.clear();
}

template<class Desc>
void stitched_vamana<Desc>::insert(const std::pair<pid_t, nid_t> &ids, const coord_t &coord,
                                   const std::vector<label_t> &F) {
  auto nid = id_map.insert(ids.first);
  assert(ids.second == nid);  // general assertion
  g.add_node(nid, node_t{coord, F});
}

template<class Desc>
template<typename Iter, class Label, class E>
void stitched_vamana<Desc>::insert2(Iter begin, Iter end, const Label &F, const E &medoid, float batch_base) {
  static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type, point_t>);
  static_assert(std::is_base_of_v<std::random_access_iterator_tag,
                                  typename std::iterator_traits<Iter>::iterator_category>);

  const size_t n = std::distance(begin, end);
  if (n == 0) return;
  assert(F.size() == n);

  // std::random_device rd;
  // auto perm = cm::random_permutation(n /*, rd()*/);
  auto rand_seq = util::delayed_seq(n, [&](size_t i) -> decltype(auto) {
    return *(begin + i /*perm[i]*/);
  });

  size_t cnt_skip = 0;
  assert(!g.empty());

  size_t batch_begin = 0, batch_end = cnt_skip;
  size_t batch_step = 0, size_limit = std::max<size_t>(n * 0.02, 20000);
  // float progress = 0.0;

  while (batch_end < n) {
    batch_begin = batch_end;
    batch_step = std::min((size_t)std::ceil(batch_step*batch_base+1), size_limit);
    batch_end = std::min<size_t>(n, batch_begin+batch_step);

    // std::cerr << "(batch_begin, batch_end)" << batch_begin << " " << batch_end << '\n';

    util::debug_output("Batch insertion: [%u, %u)\n", batch_begin, batch_end);
    // insert_batch_impl(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end);
    auto subrange = std::ranges::subrange(F.begin() + batch_begin, F.begin() + batch_end);
    std::vector new_labels(subrange.begin(), subrange.end());
    insert_batch_impl2(rand_seq.begin() + batch_begin, rand_seq.begin() + batch_end, new_labels, medoid/*, filtered*/);
  }
}

template<class Desc>
template<typename Iter, class Label>
void stitched_vamana<Desc>::insert_batch_impl(Iter begin, Iter end, const Label &F/*, bool filtered*/) {
  const size_t batch_size = std::distance(begin, end);
  assert(F.size() == batch_size);
  seq<nid_t> nids(batch_size);

  // per_visited.resize(batch_size);
  // per_eval.resize(batch_size);
  // per_size_C.resize(batch_size);

  // before the insertion, prepare the needed data
  // `nids[i]` is the nid of the node corresponding to the i-th
  // point to insert in the batch, associated with level[i]
  id_map.insert(util::delayed_seq(batch_size, [&](size_t i) {
    return (begin + i)->get_id();
  }));

  cm::parallel_for(0, batch_size, [&](uint32_t i) {
    nids[i] = id_map.get_nid((begin + i)->get_id());
  });

  g.add_nodes(util::delayed_seq(batch_size, [&](size_t i) {
    // GUARANTEE: begin[*].get_coord is only invoked for assignment once
    return std::pair{nids[i], node_t{(begin + i)->get_coord(), F[i]}};
    // return std::pair{nids[i], node_t{(begin + i)->get_coord(), *(F.begin() + i)}};
  }));

  // below we (re)generate edges incident to nodes in the current batch
  // add adges from the new points
  seq<seq<std::pair<nid_t, edge>>> edge_added(batch_size);
  seq<std::pair<nid_t, seq_edge>> nbh_forward(batch_size);

  cm::parallel_for(0, batch_size, [&](size_t i) {
    const nid_t u = nids[i];

    // auto &eps_u = entrance;
    search_control sctrl;  // TODO: use designated initializers in C++20
    sctrl.log_per_stat = i;
    // sctrl.filtered = filtered;
    // sctrl.searching = false;
    auto eps = F[i] | std::views::transform([&](label_t l){return entrance.at(l);});
    seq_conn res =
        algo::beamSearch(gen_f_nbhs(), gen_f_dist(u), get_f_label(), eps, L, F[i], sctrl);

    prune_control pctrl;  // TODO: use designated intializers in C++20
    pctrl.alpha = alpha;
    seq_conn conn_u =
        algo::prune_heuristic(std::move(res), get_deg_bound(), gen_f_nbhs(), gen_f_dist(u), pctrl);
    // record the edge for the backward insertion later
    auto &edge_cur = edge_added[i];
    edge_cur.clear();
    edge_cur.reserve(conn_u.size());
    for (const auto &[d, v] : conn_u) {
      edge_cur.emplace_back(v, edge{d, u});
    }

    // store for batch insertion
    nbh_forward[i] = {u, edge_cast(std::move(conn_u))};
  });
  util::debug_output("Adding forward edges\n");
  g.set_edges(std::move(nbh_forward));

  // now we add edges in the other direction
  auto edge_added_flatten = util::flatten(std::move(edge_added));
  auto edge_added_grouped = util::group_by_key(std::move(edge_added_flatten));

  // TODO: use std::remove_cvref in C++20
  using agent_t = std::remove_cv_t<std::remove_reference_t<decltype(g.get_edges(nid_t()))>>;
  seq<std::pair<nid_t, agent_t>> nbh_backward(edge_added_grouped.size());

  cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j) {
    nid_t v = edge_added_grouped[j].first;
    auto &nbh_v_add = edge_added_grouped[j].second;

    auto edge_agent_v = g.get_edges(v);
    auto edge_v = util::to<seq_edge>(std::move(edge_agent_v));
    // auto edge_v =
    //     util::to<seq_edge>(std::move(edge_agent_v) | std::views::filter([&](const edge &e) {
    //                          return id_map.contain_nid(e.u);
    //                        }));
    edge_v.insert(edge_v.end(), std::make_move_iterator(nbh_v_add.begin()),
                  std::make_move_iterator(nbh_v_add.end()));

    seq_conn conn_v = algo::prune_simple(conn_cast(std::move(edge_v)), get_deg_bound());
    edge_agent_v = edge_cast(conn_v);
    nbh_backward[j] = {v, std::move(edge_agent_v)};
  });
  util::debug_output("Adding backward edges\n");
  g.set_edges(std::move(nbh_backward));

  // finally, update the entrances
  // entrance.insert(entrance.end(), nids.begin(), nids.end());
}

template<class Desc>
template<typename Iter, class Label, class E>
void stitched_vamana<Desc>::insert_batch_impl2(Iter begin, Iter end, const Label &F, const E &medoid) {
  const size_t batch_size = std::distance(begin, end);
  assert(F.size() == batch_size);
  seq<nid_t> nids(batch_size);

  // before the insertion, prepare the needed data
  // `nids[i]` is the nid of the node corresponding to the i-th
  // point to insert in the batch, associated with level[i]
  id_map.insert(util::delayed_seq(batch_size, [&](size_t i) {
    return (begin + i)->get_id();
  }));

  cm::parallel_for(0, batch_size, [&](uint32_t i) {
    nids[i] = id_map.get_nid((begin + i)->get_id());
  });

  g.add_nodes(util::delayed_seq(batch_size, [&](size_t i) {
    // GUARANTEE: begin[*].get_coord is only invoked for assignment once
    return std::pair{nids[i], node_t{(begin + i)->get_coord(), F[i]}};
    // return std::pair{nids[i], node_t{(begin + i)->get_coord(), *(F.begin() + i)}};
  }));

  // below we (re)generate edges incident to nodes in the current batch
  // add adges from the new points
  seq<seq<std::pair<nid_t, edge>>> edge_added(batch_size);
  seq<std::pair<nid_t, seq_edge>> nbh_forward(batch_size);

  cm::parallel_for(0, batch_size, [&](size_t i) {
    const nid_t u = nids[i];

    search_control sctrl;  // TODO: use designated initializers in C++20
    sctrl.log_per_stat = i;
    // sctrl.filtered = filtered;
    // sctrl.searching = false;
    seq_conn res =
        algo::beamSearch3(gen_f_nbhs(), gen_f_dist(u), get_f_label(), medoid, L, F[i], sctrl);

    prune_control pctrl;  // TODO: use designated intializers in C++20
    pctrl.alpha = alpha;
    seq_conn conn_u =
        algo::prune_heuristic(std::move(res), get_deg_bound(), gen_f_nbhs(), gen_f_dist(u), pctrl);
    // record the edge for the backward insertion later
    auto &edge_cur = edge_added[i];
    edge_cur.clear();
    edge_cur.reserve(conn_u.size());
    for (const auto &[d, v] : conn_u) {
      edge_cur.emplace_back(v, edge{d, u});
    }

    // store for batch insertion
    nbh_forward[i] = {u, edge_cast(std::move(conn_u))};
  });
  util::debug_output("Adding forward edges\n");
  g.set_edges(std::move(nbh_forward));

  // now we add edges in the other direction
  auto edge_added_flatten = util::flatten(std::move(edge_added));
  auto edge_added_grouped = util::group_by_key(std::move(edge_added_flatten));

  // TODO: use std::remove_cvref in C++20
  using agent_t = std::remove_cv_t<std::remove_reference_t<decltype(g.get_edges(nid_t()))>>;
  seq<std::pair<nid_t, agent_t>> nbh_backward(edge_added_grouped.size());

  cm::parallel_for(0, edge_added_grouped.size(), [&](size_t j) {
    nid_t v = edge_added_grouped[j].first;
    auto &nbh_v_add = edge_added_grouped[j].second;

    auto edge_agent_v = g.get_edges(v);
    auto edge_v = util::to<seq_edge>(std::move(edge_agent_v));
    // auto edge_v =
    //     util::to<seq_edge>(std::move(edge_agent_v) | std::views::filter([&](const edge &e) {
    //                          return id_map.contain_nid(e.u);
    //                        }));
    edge_v.insert(edge_v.end(), std::make_move_iterator(nbh_v_add.begin()),
                  std::make_move_iterator(nbh_v_add.end()));

    seq_conn conn_v = algo::prune_simple(conn_cast(std::move(edge_v)), get_deg_bound());
    edge_agent_v = edge_cast(conn_v);
    nbh_backward[j] = {v, std::move(edge_agent_v)};
  });
  util::debug_output("Adding backward edges\n");
  g.set_edges(std::move(nbh_backward));
}

template<class Desc>
template<class Seq>
Seq stitched_vamana<Desc>::search(const coord_t &cq, uint32_t k, uint32_t ef,
                                  const std::vector<label_t> &F, const search_control &ctrl) const {
  // seq<nid_t> eps = entrance;
  // auto nbhs = beamSearch(gen_f_nbhs(), gen_f_dist(cq), eps, ef, ctrl);
  auto eps = F | std::views::transform([&](label_t l){return entrance.at(l);});
  auto nbhs = algo::beamSearch(gen_f_nbhs(), gen_f_dist(cq), get_f_label(), eps, ef, F, ctrl);

  cm::sort(nbhs.begin(), nbhs.end());
  nbhs = algo::prune_simple(std::move(nbhs), k /*, ctrl*/);  // TODO: set ctrl

  using result_t = typename Seq::value_type;
  static_assert(util::is_direct_list_initializable_v<result_t, dist_t, pid_t>);

  Seq res(nbhs.size());
  cm::parallel_for(0, nbhs.size(), [&](size_t i) {
    const auto &nbh = nbhs[i];
    res[i] = result_t{nbh.d, id_map.get_pid(nbh.u)};
  });

  return res;
}

}  // namespace ANN

#endif  // _ANN_ALGO_STITCHED_VAMANA_HPP
