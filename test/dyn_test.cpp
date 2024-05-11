#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "algo/algo.hpp"
#include "algo/vamana.hpp"
#include "benchUtils.h"
// #include "cpam.hpp"
#include "dist.hpp"
#include "graph/adj.hpp"
#include "parlay.hpp"
#include "parse_points.hpp"
#include "util/intrin.hpp"

using ANN::vamana;

parlay::sequence<size_t> per_visited;
parlay::sequence<size_t> per_eval;
parlay::sequence<size_t> per_size_C;

namespace ANN::external {

  auto def_custom_tag() {
    return custom_tag_parlay{};
  }

}  // namespace ANN::external

template<typename T>
point_converter_default<T> to_point;

template<typename T>
class gt_converter {
 public:
  using type = parlay::sequence<T>;
  template<typename Iter>
  type operator()([[maybe_unused]] uint32_t id, Iter begin, Iter end) {
    using type_src = typename std::iterator_traits<Iter>::value_type;
    static_assert(std::is_convertible_v<type_src, T>, "Cannot convert to the target type");

    const uint32_t n = std::distance(begin, end);

    // T *gt = new T[n];
    auto gt = parlay::sequence<T>(n);
    for (uint32_t i = 0; i < n; ++i) gt[i] = *(begin + i);
    return gt;
  }
};

template<class DescLegacy>
struct desc {
  using point_t = point<typename DescLegacy::type_elem>;
  using coord_t = typename point_t::coord_t;
  using label_t = typename point_t::label_t;
  using dist_t = float;
  static dist_t distance(const coord_t &cu, const coord_t &cv, uint32_t dim) {
    return DescLegacy::distance(cu, cv, dim);
  }

  template<typename Nid, class Ext, class Edge>
  // using graph_t = ANN::graph::adj_seq<Nid,Ext,Edge>;
  using graph_t = ANN::graph::adj_map<Nid, Ext, Edge>;

  template<typename Nid, class Ext, class Edge>
  using graph_aux = ANN::graph::adj_map<Nid, Ext, Edge>;
};

// template<class DescLegacy>
// struct desc_cpam : desc<DescLegacy> {
//   template<typename Nid, class Ext, class Edge = Nid>
//   using graph_t = graph_cpam<Nid, Ext, Edge>;

//   template<typename Nid, class Ext, class Edge = Nid>
//   using graph_aux = graph_cpam<Nid, Ext, Edge>;
// };

// Visit all the vectors in the given 2D array of points
// This triggers the page fetching if the vectors are mmap-ed
template<class T>
void visit_point(const T &array, size_t dim0, size_t dim1) {
  parlay::parallel_for(0, dim0, [&](size_t i) {
    const auto &a = array[i];
    [[maybe_unused]] volatile auto elem = a.get_coord()[0];
    for (size_t j = 1; j < dim1; ++j) elem = a.get_coord()[j];
  });
}

template<class G>
void print_stat(const G &g) {
  puts("#vertices         edges    avg. deg");
  size_t cnt_vertex = g.num_nodes();
  size_t cnt_degree = g.num_edges();
  printf("%4lu %16lu %10.2f\n", cnt_vertex, cnt_degree, float(cnt_degree) / cnt_vertex);
}
/*
// assume the snapshots are only of increasing insertions
template<class R>
void print_stat_snapshots(const R &snapshots)
{
        const auto &fin = snapshots.back();
        std::vector<std::unordered_set<uint64_t>> cnts_vtx(fin.get_height()+1);
        std::vector<std::unordered_map<uint64_t,size_t>> cnts_edge(fin.get_height()+1);
        // count the vertices and the edges
        for(const auto &g : snapshots)
        {
                auto sum_edges = [](const auto &m){
                        return std::transform_reduce(
                                m.begin(), m.end(),
                                size_t(0),
                                std::plus<size_t>{},
                                // [](uint64_t eid){return size_t(eid&0xffff);}
                                [](std::pair<uint64_t,size_t> p){return p.second;}
                        );
                };

                const uint32_t height = g.get_height();
                printf("Highest level: %u\n", height);
                puts("level     #vertices      edges    avg. deg   ref#vtx   ref#edge");
                for(uint32_t i=0; i<=height; ++i)
                {
                        const uint32_t l = height-i;
                        size_t cnt_vertex = g.num_nodes(l);
                        size_t cnt_degree = g.num_edges(l);
                        auto refs = g.collect_refs(l);

                        // TODO: use std::transform_view in C++20
                        for(auto [key,val] : refs)
                        {
                                cnts_vtx[l].insert(key);
                                uint64_t ptr_el = val>>16;
                                size_t size_el = val&0xffff;
                                auto &old_size = cnts_edge[l][ptr_el];
                                if(old_size<size_el)
                                        old_size = size_el;
                        }

                        size_t ref_vtx = cnts_vtx[l].size();
                        size_t ref_edge = sum_edges(cnts_edge[l]);
                        printf("#%2u: %14lu %10lu %10.2f %10lu %10lu\n",
                                l, cnt_vertex, cnt_degree, float(cnt_degree)/cnt_vertex,
                                ref_vtx, ref_edge
                        );
                }
        }
        // sample 100 vertices and observe the historical changes of their edge lists
        std::mt19937 gen(1206);
        std::uniform_int_distribution<uint32_t> distrib(0, fin.num_nodes(0));
        std::vector<uint32_t> samples(100);
        for(uint32_t &sample : samples)
        {
                sample = distrib(gen);
                printf("%u ", sample);
        }
        putchar('\n');

        for(size_t i=0; i<snapshots.size(); ++i)
        {
                const auto &s = snapshots[i];
                for(uint32_t sample : samples)
                {
                        if(sample>=s.num_nodes(0))
                        {
                                // printf("* ");
                                printf("0 ");
                                continue;
                        }
                        if(i==0 || sample>=snapshots[i-1].num_nodes(0))
                        {
                                // printf("+%lu,-0 ", s.get_nbhs(sample).size());
                                printf("%lu ", s.get_nbhs(sample).size());
                                continue;
                        }

                        auto calc_diff = [](auto agent_last, auto agent_curr){
                                using edge_t = typename decltype(agent_last)::value_type;
                                auto last = ANN::util::to<parlay::sequence<edge_t>>(agent_last);
                                auto curr = ANN::util::to<parlay::sequence<edge_t>>(agent_curr);
                                std::sort(last.begin(), last.end());
                                std::sort(curr.begin(), curr.end());
                                parlay::sequence<edge_t> comm;
                                std::set_intersection(
                                        last.begin(), last.end(),
                                        curr.begin(), curr.end(),
                                        std::inserter(comm, comm.end())
                                );
                                return std::make_pair(
                                        curr.size()-comm.size(),
                                        last.size()-comm.size()
                                );
                        };

                        auto diff = calc_diff(
                                snapshots[i-1].get_nbhs(sample),
                                s.get_nbhs(sample)
                        );
                        // printf("+%lu,-%lu ", diff.first, diff.second);
                        printf("%lu ", diff.first);
                }
                putchar('\n');
        }
}
*/

template<class G, class Seq>
auto find_nbhs(const G &g, const Seq &q, uint32_t k, uint32_t ef) {
  const size_t cnt_query = q.size();
  per_visited.resize(cnt_query);
  per_eval.resize(cnt_query);
  per_size_C.resize(cnt_query);

  using seq_result = parlay::sequence<typename G::result_t>;
  parlay::sequence<seq_result> res(cnt_query);
  auto search = [&] {
    parlay::parallel_for(0, cnt_query, [&](size_t i) {
      ANN::algo::search_control ctrl{};
      ctrl.log_per_stat = i;
      // ctrl.beta = beta;
      res[i] = g.template search<seq_result>(q[i].get_coord(), k, ef, ctrl);
    });
  };

  puts("Warmup");
  search();

  parlay::internal::timer t;
  const uint32_t rounds = 3;
  for (uint32_t i = 0; i < rounds; ++i) search();
  const double time_query = t.next_time() / rounds;
  const double qps = cnt_query / time_query;
  printf("Find neighbors: %.4f s, %e kqps\n", time_query, qps / 1000);

  printf("# visited: %lu\n", parlay::reduce(per_visited, parlay::addm<size_t>{}));
  printf("# eval: %lu\n", parlay::reduce(per_eval, parlay::addm<size_t>{}));
  printf("size of C: %lu\n", parlay::reduce(per_size_C, parlay::addm<size_t>{}));

  parlay::sort_inplace(per_visited);
  parlay::sort_inplace(per_eval);
  parlay::sort_inplace(per_size_C);
  const double tail_ratio[] = {0.9, 0.99, 0.999};
  for (size_t i = 0; i < sizeof(tail_ratio) / sizeof(*tail_ratio); ++i) {
    const auto r = tail_ratio[i];
    const uint32_t tail_index = r * cnt_query;
    printf("%.4f tail stat (at %u):\n", r, tail_index);

    printf("\t# visited: %lu\n", per_visited[tail_index]);
    printf("\t# eval: %lu\n", per_eval[tail_index]);
    printf("\tsize of C: %lu\n", per_size_C[tail_index]);
  }

  return res;
}

template<class G, class Seq, typename L>
auto find_nbhs(const G &g, const Seq &q, uint32_t k, uint32_t ef,
               const std::vector<std::vector<L>> &F) {
  const size_t cnt_query = q.size();
  per_visited.resize(cnt_query);
  per_eval.resize(cnt_query);
  per_size_C.resize(cnt_query);

  using seq_result = parlay::sequence<typename G::result_t>;
  parlay::sequence<seq_result> res(cnt_query);
  auto search = [&] {
    parlay::parallel_for(0, cnt_query, [&](size_t i) {
      ANN::algo::search_control ctrl{};
      ctrl.log_per_stat = i;
      // ctrl.beta = beta;
      // res[i] = g.template search<seq_result>(q[i].get_coord(), k, ef, ctrl);
      res[i] = g.template search<seq_result>(q[i].get_coord(), k, ef, F[i], ctrl);
    });
  };

  puts("Warmup");
  search();

  parlay::internal::timer t;
  const uint32_t rounds = 3;
  for (uint32_t i = 0; i < rounds; ++i) search();
  const double time_query = t.next_time() / rounds;
  const double qps = cnt_query / time_query;
  printf("Find neighbors: %.4f s, %e kqps\n", time_query, qps / 1000);

  printf("# visited: %lu\n", parlay::reduce(per_visited, parlay::addm<size_t>{}));
  printf("# eval: %lu\n", parlay::reduce(per_eval, parlay::addm<size_t>{}));
  printf("size of C: %lu\n", parlay::reduce(per_size_C, parlay::addm<size_t>{}));

  parlay::sort_inplace(per_visited);
  parlay::sort_inplace(per_eval);
  parlay::sort_inplace(per_size_C);
  const double tail_ratio[] = {0.9, 0.99, 0.999};
  for (size_t i = 0; i < sizeof(tail_ratio) / sizeof(*tail_ratio); ++i) {
    const auto r = tail_ratio[i];
    const uint32_t tail_index = r * cnt_query;
    printf("%.4f tail stat (at %u):\n", r, tail_index);

    printf("\t# visited: %lu\n", per_visited[tail_index]);
    printf("\t# eval: %lu\n", per_eval[tail_index]);
    printf("\tsize of C: %lu\n", per_size_C[tail_index]);
  }

  return res;
}

template<class U, class Seq, class Point>
auto CalculateOneKnn(const Seq &data, const Point &q, uint32_t dim, uint32_t k) {
  static_assert(std::is_same_v<Point, typename U::point_t>);

  using pid_t = typename U::point_t::id_t;
  std::priority_queue<std::pair<float, pid_t>> top_candidates;
  float lower_bound = std::numeric_limits<float>::min();

  for (size_t i = 0; i < data.size(); ++i) {
    const auto &u = data[i];
    float dist = U::distance(u.get_coord(), q.get_coord(), dim);

    // only keep the top k
    if (top_candidates.size() < k || dist < lower_bound) {
      top_candidates.emplace(dist, u.get_id());
      if (top_candidates.size() > k) top_candidates.pop();
      lower_bound = top_candidates.top().first;
    }
  }

  parlay::sequence<pid_t> knn;
  while (!top_candidates.empty()) {
    knn.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  std::reverse(knn.begin(), knn.end());
  return knn;
}

template<class U, class Seq, class Point, typename L>
auto CalculateOneKnn(const Seq &data, const Point &q, uint32_t dim, uint32_t k,
                     const std::vector<std::vector<L>> &base_labels,
                     const std::vector<L> &query_label) {
  static_assert(std::is_same_v<Point, typename U::point_t>);
  assert(data.size() == base_labels.size());

  using pid_t = typename U::point_t::id_t;
  std::priority_queue<std::pair<float, pid_t>> top_candidates;
  float lower_bound = std::numeric_limits<float>::min();

  for (size_t i = 0; i < data.size(); ++i) {
    const auto &u = data[i];  // id, point
    std::vector<L> inter;
    std::set_intersection(base_labels[i].begin(), base_labels[i].end(), query_label.begin(),
                          query_label.end(), std::back_inserter(inter));

    if (inter.size() == 0) continue;
    float dist = U::distance(u.second.get_coord(), q.get_coord(), dim);

    // only keep the top k
    if (top_candidates.size() < k || dist < lower_bound) {
      top_candidates.emplace(dist, u.first);
      if (top_candidates.size() > k) {
        top_candidates.pop();
      }
      lower_bound = top_candidates.top().first;
    }
  }

  parlay::sequence<pid_t> knn;
  while (!top_candidates.empty()) {
    knn.emplace_back(top_candidates.top().second);
    top_candidates.pop();
  }
  std::reverse(knn.begin(), knn.end());
  return knn;
}

template<class U, class S1, class S2>
auto ConstructKnng(const S1 &data, const S2 &qs, uint32_t dim, uint32_t k) {
  using pid_t = typename U::point_t::id_t;
  parlay::sequence<parlay::sequence<pid_t>> res(qs.size());
  parlay::parallel_for(0, qs.size(),
                       [&](size_t i) { res[i] = CalculateOneKnn<U>(data, qs[i], dim, k); });
  return res;
}

template<class U, class S1, class S2, typename L>
auto ConstructKnng(const S1 &data, const S2 &qs, uint32_t dim, uint32_t k,
                   const std::vector<std::vector<L>> &base_labels,
                   const std::vector<std::vector<L>> &query_labels) {
  using pid_t = typename U::point_t::id_t;
  parlay::sequence<parlay::sequence<pid_t>> res(qs.size());
  parlay::parallel_for(0, qs.size(), [&](size_t i) {
    res[i] = CalculateOneKnn<U>(data, qs[i], dim, k, base_labels, query_labels[i]);
  });
  return res;
}

template<class S1, class S2, class S3>
void calc_recall(const S1 &q, const S2 &res, const S3 &gt, uint32_t k) {
  const size_t cnt_query = q.size();
  //	uint32_t cnt_all_shot = 0;
  std::vector<uint32_t> result(k + 1);
  printf("measure recall@%u on %lu queries\n", k, cnt_query);
  for (uint32_t i = 0; i < cnt_query; ++i) {
    uint32_t cnt_shot = 0;
    for (uint32_t j = 0; j < k; ++j) {
      // std::cout << res[i][j].pid << ", " << gt[i][j] << '\n';
      if (std::find_if(res[i].begin(), res[i].end(),
                       [&](const auto &p) { return p.pid == gt[i][j]; }) !=
          res[i].end())  // TODO: fix naming
      {
        cnt_shot++;
      }
    }
    result[cnt_shot]++;
  }
  size_t total_shot = 0;
  for (size_t i = 0; i <= k; ++i) {
    printf("%u ", result[i]);
    total_shot += result[i] * i;
  }
  putchar('\n');
  printf("recall: %.6f\n", float(total_shot) / cnt_query / k);
}

template<typename U>
void run_test(commandLine parameter)  // intend to be pass-by-value manner
{
  const char *file_in = parameter.getOptionValue("-in");
  const size_t size_init = parameter.getOptionLongValue("-init", 0);
  const size_t size_step = parameter.getOptionLongValue("-step", 0);
  size_t size_max = parameter.getOptionLongValue("-max", 0);
  const uint32_t m = parameter.getOptionIntValue("-m", 40);
  const uint32_t efc = parameter.getOptionIntValue("-efc", 60);
  const float alpha = parameter.getOptionDoubleValue("-alpha", 1);
  const float batch_base = parameter.getOptionDoubleValue("-b", 2);
  const char *file_query = parameter.getOptionValue("-q");
  const uint32_t k = parameter.getOptionIntValue("-k", 10);
  const uint32_t ef = parameter.getOptionIntValue("-ef", m * 20);
  const char *file_label_in = parameter.getOptionValue("-lb");
  const char *file_label_query = parameter.getOptionValue("-lq");

  parlay::internal::timer t("run_test:prepare", true);

  using L = typename U::point_t::label_t;
  auto [F_b, P] = load_label<L>(file_label_in, size_max);
  t.next("Load base labels");
  printf("Load %lu base points w/ labels\n", F_b.size());
  const size_t n_unique_label = P.size();

  using T = typename U::point_t::elem_t;
  auto [ps, dim] = load_point(file_in, to_point<T>, size_max);
  t.next("Load the base set");
  printf("%s: [%lu,%u]\n", file_in, ps.size(), dim);

  if (ps.size() < size_max) {
    size_max = ps.size();
    printf("size_max is corrected to %lu\n", size_max);
  }

  auto [F_q, _pq] = load_label<L>(file_label_query);
  t.next("Load query labels");
  printf("Load %lu query points w/ labels\n", F_q.size());

  auto [q, _] = load_point(file_query, to_point<T>);
  t.next("Load queries");
  printf("%s: [%lu,%u]\n", file_query, q.size(), _);

  // parallel pre-fetch
  visit_point(ps, size_max, dim);
  visit_point(q, q.size(), dim);
  t.next("Prefetch vectors");

  vamana<U> base(dim, m, efc, alpha);
  // std::vector<vamana<U>> snapshots;
  puts("Initialize base vamana");

  auto Merge = [alpha](vamana<U> from, vamana<U> &to) {
    from.g.for_each_nid([&](typename vamana<U>::nid_t nid) {
      // for (typename vamana<U>::nid_t nid : from.existed_points) {
      // std::cerr << "nid: " << nid << '\n';
      if (to.is_node_existed(nid)) {
        auto new_edges = to.g.get_edges(nid);
        auto edge_v = ANN::util::to<typename vamana<U>::seq_edge>(std::move(new_edges));
        // std::cerr << "Exist edge total before: " << edge_v.size() << '\n';
        edge_v.insert(edge_v.end(), std::make_move_iterator(new_edges.begin()),
                      std::make_move_iterator(new_edges.end()));
        // std::cerr << "Exist edge total after: " << edge_v.size() << '\n';
        // edge_v.insert(edge_v.end(), new_edges.begin(), new_edges.end());
        // typename vamana<U>::seq_conn conn_v = ANN::algo::prune_simple(
        //     to.conn_cast(std::move(edge_v)), to.get_deg_bound());
        typename vamana<U>::prune_control pctrl;
        pctrl.alpha = alpha;
        typename vamana<U>::seq_conn conn_v =
            ANN::algo::prune_heuristic(to.conn_cast(std::move(edge_v)), to.get_deg_bound(),
                                       to.gen_f_nbhs(to.g), to.gen_f_dist(nid), pctrl);
        new_edges = to.edge_cast(std::move(conn_v));
        to.g.set_edges(nid, std::move(new_edges));
      } else {
        const auto &node = from.g.get_node(nid);
        const auto &labels = node->get_label();
        const auto &coord = node->get_coord();
        to.insert(nid, coord, labels);
        auto edges = from.g.get_edges(nid);
        to.g.set_edges(nid, std::move(edges));
      }
      // }
    });
    for (const auto &ep : from.entrance) {
      to.entrance.push_back(ep);
    }
    std::sort(to.entrance.begin(), to.entrance.end());
    to.entrance.erase(std::unique(to.entrance.begin(), to.entrance.end()), to.entrance.end());
  };

  size_t idx = 0;
  for (const auto &[f, Pf] : P) {
    // std::cerr << f << " " << Pf.size() << '\n';
    // parlay::sequence<typename decltype(ps)::value_type> new_ps(Pf.size());
    parlay::sequence<std::pair<L, typename decltype(ps)::value_type>> new_ps(Pf.size());
    std::vector<std::vector<L>> base_labels(Pf.size());

    parlay::parallel_for(0, Pf.size(), [&](size_t i) {
      // new_ps[i] = *(ps.begin() + Pf[i]);
      new_ps[i] = std::make_pair(Pf[i], *(ps.begin() + Pf[i]));
      base_labels[i] = F_b[Pf[i]];
    });

    // for(size_t size_last=0, size_curr=size_init; size_curr<=std::min<size_t>(size_max,
    // new_ps.size()); size_last=size_curr, size_curr+=size_step)
    // {
    // printf("Increasing size from %lu to %lu\n", size_last, size_curr);

    puts("Insert points");
    printf("Inserted points w/ label: %lu/%lu\n", ++idx, n_unique_label);
    parlay::internal::timer t("run_test:insert", true);

    // auto ins_begin = new_ps.begin() + size_last;
    // auto ins_end = new_ps.begin() + size_curr;
    auto ins_begin = new_ps.begin();
    auto ins_end = new_ps.end();

    // g.insert(ins_begin, ins_end, batch_base);
    // g.insert(ins_begin, ins_end, F_b, batch_base);

    // if (size_last == 0) {
    // 	base.insert(ins_begin, ins_end, F_b, batch_base);
    // 	// is_first_insert = false;
    // } else {
    // 	vamana<U> g(dim, m, efc, alpha);
    // 	g.insert(ins_begin, ins_end, F_b, batch_base);
    // 	Merge(g, base);
    // 	// base.insert(ins_begin, ins_end, F_b, batch_base);
    // }

    vamana<U> g(dim, m, efc, alpha);
    g.insert(ins_begin, ins_end, base_labels, batch_base);
    Merge(g, base);
    t.next("Finish insertion");

    // snapshots.push_back(g);

    puts("Collect statistics");
    print_stat(base);

    // puts("Search for neighbors");
    // auto res = find_nbhs(base, q, k, ef, F_q);

    // puts("Generate groundtruth");
    // // auto baseset = parlay::make_slice(ps.begin(), ins_end);
    // // auto gt = ConstructKnng<U>(baseset, q, dim, k);

    // // auto baseset = parlay::make_slice(new_ps.begin(), ins_end);
    // // std::vector label_slice(F_b.begin(), F_b.begin() + size_curr);
    // // auto gt = ConstructKnng<U>(baseset, q, dim, k, label_slice, F_q);

    // auto gt = ConstructKnng<U>(new_ps, q, dim, k, base_labels, F_q);

    // puts("Compute recall");
    // calc_recall(q, res, gt, k);

    // puts("---");
    // }
  }

  puts("Search for neighbors");
  auto res = find_nbhs(base, q, k, ef, F_q);

  puts("Generate groundtruth");
  // auto baseset = parlay::make_slice(ps.begin(), ins_end);
  // auto gt = ConstructKnng<U>(baseset, q, dim, k);

  // auto baseset = parlay::make_slice(new_ps.begin(), ins_end);
  // std::vector label_slice(F_b.begin(), F_b.begin() + size_curr);
  // auto gt = ConstructKnng<U>(baseset, q, dim, k, label_slice, F_q);

  parlay::sequence<std::pair<L, typename decltype(ps)::value_type>> new_ps(ps.size());
  parlay::parallel_for(0, ps.size(),
                       [&](size_t i) { new_ps[i] = std::make_pair(i, *(ps.begin() + i)); });
  assert(new_ps.size() == F_b.size());
  auto gt = ConstructKnng<U>(new_ps, q, dim, k, F_b, F_q);

  puts("Compute recall");
  calc_recall(q, res, gt, k);

  puts("---");
  // print_stat_snapshots(snapshots);
}

int main(int argc, char **argv) {
  for (int i = 0; i < argc; ++i) printf("%s ", argv[i]);
  putchar('\n');

  commandLine parameter(argc, argv,
                        "-type <elemType> -dist <distance>"
                        "-ml <m_l> -m <m> -efc <ef_construction> -alpha <alpha> "
                        "-in <baseset> -q <queries> "
                        "-init <init_size> -step <step_size> -max <max_size>"
                        "-k <recall@k> -ef <ef_query> [-beta <beta>,...]");

  const char *dist_func = parameter.getOptionValue("-dist");
  auto run_test_helper = [&](auto type) {  // emulate a generic lambda in C++20
    using T = decltype(type);
    if (!strcmp(dist_func, "L2")) run_test<desc<descr_l2<T>>>(parameter);
    /*
    else if(!strcmp(dist_func,"angular"))
            run_test<desc<descr_ang<T>>>(parameter);
    else if(!strcmp(dist_func,"ndot"))
            run_test<desc<descr_ndot<T>>>(parameter);
    */
    else
      throw std::invalid_argument("Unsupported distance type");
  };

  const char *type = parameter.getOptionValue("-type");

  if (!strcmp(type, "float")) {
    run_test_helper(float{});
  }
  // if(!strcmp(type,"uint8"))
  // 	run_test_helper(uint8_t{});
  /*
  else if(!strcmp(type,"int8"))
          run_test_helper(int8_t{});
  else if(!strcmp(type,"float"))
          run_test_helper(float{});
  */
  else {
    throw std::invalid_argument("Unsupported element type");
  }
  return 0;
}
