#ifndef __DIST_HPP__
#define __DIST_HPP__

#include <type_traits>

#include "NSGDist.h"

template<typename T>
class descr_ang {
  using promoted_type = std::conditional_t<std::is_integral_v<T> && sizeof(T) <= 4,
                                           std::conditional_t<sizeof(T) == 4, int64_t, int32_t>, float>;

 public:
  typedef T type_elem;
  typedef point<T> type_point;
  typedef typename type_point::coord_t type_coord;
  static float distance(const type_coord &uc, const type_coord &vc, uint32_t dim) {
    promoted_type dot = 0, nu = 0, nv = 0;
    for (uint32_t i = 0; i < dim; ++i) {
      nu += promoted_type(uc[i]) * uc[i];
      nv += promoted_type(vc[i]) * vc[i];
      dot += promoted_type(uc[i]) * vc[i];
    }
    return 1 - dot / (sqrtf(nu) * sqrtf(nv));
  }

  static auto get_id(const type_point &u) {
    return u.id;
  }
};

template<typename T>
class descr_ndot {
  using promoted_type = std::conditional_t<std::is_integral_v<T> && sizeof(T) <= 4,
                                           std::conditional_t<sizeof(T) == 4, int64_t, int32_t>, float>;

 public:
  typedef T type_elem;
  typedef point<T> type_point;
  typedef typename type_point::coord_t type_coord;
  static float distance(const type_coord &uc, const type_coord &vc, uint32_t dim) {
    promoted_type dot = 0;
    for (uint32_t i = 0; i < dim; ++i) dot += promoted_type(uc[i]) * vc[i];
    return -float(dot);
  }

  static auto get_id(const type_point &u) {
    return u.id;
  }
};

template<typename T>
class descr_l2 {
  using promoted_type = std::conditional_t<std::is_integral_v<T> && sizeof(T) <= 4,
                                           std::conditional_t<sizeof(T) == 4, int64_t, int32_t>, float>;

 public:
  typedef T type_elem;
  typedef point<T> type_point;
  typedef typename type_point::coord_t type_coord;
  static float distance(const type_coord &uc, const type_coord &vc, uint32_t dim) {
    if constexpr (std::is_integral_v<T>) {
      promoted_type sum = 0;
      for (uint32_t i = 0; i < dim; ++i) {
        const auto d = promoted_type(uc[i]) - vc[i];
        sum += d * d;
      }
      return sum;
    } else {
      efanna2e::DistanceL2 distfunc;
      return distfunc.compare(uc, vc, dim);
    }
  }

  static auto get_id(const type_point &u) {
    return u.id;
  }
};

#endif  // __DIST_HPP__
