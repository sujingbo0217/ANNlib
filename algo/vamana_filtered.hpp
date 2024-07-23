#ifndef _ANN_ALGO_FILTERED_VAMANA_HPP
#define _ANN_ALGO_FILTERED_VAMANA_HPP

#include "algo/vamana_stitched.hpp"

namespace ANN {

  template<class Desc>
  class filtered_vamana : public stitched_vamana<Desc> {

   public:
    filtered_vamana(uint32_t dim, uint32_t R, uint32_t L, float alpha)
        : stitched_vamana<Desc>(dim, R, L, alpha) {}

  };

}  // namespace ANN

#endif  // _ANN_ALGO_FILTERED_VAMANA_HPP
