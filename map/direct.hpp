#ifndef _ANN_MAP_DIRECT_HPP
#define _ANN_MAP_DIRECT_HPP

#include <iterator>
#include <type_traits>
#include <utility>

#include "map.hpp"

namespace ANN::map {

  template<typename Pid, typename Nid>
  class direct {
   public:
    template<typename Iter>
    void insert(Iter begin, Iter end) {
      (void)begin, (void)end;
    }

    template<class Ctr>
    void insert(Ctr &&c) {
      if constexpr (std::is_rvalue_reference_v<Ctr &&>) {
        insert(std::make_move_iterator(c.begin()), std::make_move_iterator(c.end()));
      } else
        insert(c.begin(), c.end());
    }

    Nid insert(const Pid &pid) {
      return Nid(pid);
    }

    Nid insert(Pid &&pid) {
      return Nid(std::move(pid));
    }

    Pid get_pid(Nid nid) const {
      static_assert(std::is_convertible_v<Nid, Pid>);
      return Pid(nid);
    }

    Nid get_nid(const Pid &pid) const {
      static_assert(std::is_convertible_v<Pid, Nid>);
      return Nid(pid);
    }
  };

}  // namespace ANN::map

#endif  // _ANN_MAP_DIRECT_HPP
