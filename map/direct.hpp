#ifndef _ANN_MAP_DIRECT_HPP
#define _ANN_MAP_DIRECT_HPP

#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <type_traits>
#include <unordered_map>

#include "map.hpp"
#include "util/seq.hpp"

namespace ANN::map {

  template<typename Pid, typename Nid>
  class direct {
    std::unordered_map<Nid, Pid> mapping;
    // std::unordered_map<Pid, Nid> rev_mapping;

    static Nid f(const Pid &pid) {
      static_assert(std::is_convertible_v<size_t, Nid>);
      return std::hash<Pid>{}(pid);
    }

   public:
    template<typename Iter>
    void insert(Iter begin, Iter end) {
      // use mapping.insert_range() in C++23
      const auto n = std::distance(begin, end);

      auto ps = util::delayed_seq(n, [&](size_t i) {
        auto &&pid = *(begin + i);
        return std::pair<Nid, Pid>(f(pid), std::forward<decltype(pid)>(pid));
      });
      mapping.insert(ps.begin(), ps.end());

      // auto rps = util::delayed_seq(n, [&](size_t i) {
      //   auto &&pid = *(begin + i);
      //   return std::pair<Pid, Nid>(std::forward<decltype(pid)>(pid), f(pid));
      // });
      // rev_mapping.insert(rps.begin(), rps.end());
    }

    template<class Ctr>
    void insert(Ctr &&c) {
      if constexpr (std::is_rvalue_reference_v<Ctr &&>) {
        insert(std::make_move_iterator(c.begin()), std::make_move_iterator(c.end()));
      } else {
        insert(c.begin(), c.end());
      }
    }

    Nid insert(const Pid &pid) {
      // rev_mapping.insert({pid, f(pid)});
      return mapping.insert({f(pid), pid}).first->first;
    }

    Nid insert(Pid &&pid) {
      // rev_mapping.insert({std::move(pid), f(pid)});
      return mapping.insert({f(pid), std::move(pid)}).first->first;
    }

    template<typename Iter>
    void erase(Iter begin, Iter end) {
      for (auto it = begin; it != end; ++it) {
				mapping.erase(f(*it));
				// rev_mapping.erase(*it);
			}
    }

    void erase(Nid nid) {
      mapping.erase(nid);
			// rev_mapping.erase(get_pid(nid));
    }

    Pid get_pid(Nid nid) const {
      static_assert(std::is_convertible_v<Nid, Pid>);
      return Pid(nid);
    }

    Nid get_nid(const Pid &pid) const {
      static_assert(std::is_convertible_v<Pid, Nid>);
      return Nid(pid);
    }

    Nid front_nid() const {
      return mapping.begin()->first;
    };

    // TODO: consider to remove it
    std::optional<Nid> find_nid(const Pid &pid) const {
      auto it = mapping.find(f(pid));
      if (it == mapping.end()) return std::nullopt;
      return {it->first};
    }

    // TODO: use a more unambiguous name
    bool contain_nid(Nid nid) const {
      return mapping.contains(nid);
    }

    bool has_node(const Nid &nid) {
      return (mapping.find(nid) != mapping.end());
    }

    std::unordered_map<Pid, Nid> get_map() const {
      return mapping;
    }
  };

}  // namespace ANN::map

#endif  // _ANN_MAP_DIRECT_HPP
