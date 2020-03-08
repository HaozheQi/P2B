#include "ball_query_score.h"
#include "utils.h"

void query_ball_point_score_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const float *score, float *unique_score);


at::Tensor ball_query_score(at::Tensor new_xyz, at::Tensor xyz, at::Tensor score, const float radius,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(score);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(score);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
    CHECK_CUDA(score);
  }

  at::Tensor unique_score =
      torch::zeros({new_xyz.size(0), new_xyz.size(1)},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Float));

  if (new_xyz.type().is_cuda()) {
    query_ball_point_score_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    radius, nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), score.data<float>(), unique_score.data<float>());
  } else {
    AT_CHECK(false, "CPU not supported");
  }

  return unique_score;
}
