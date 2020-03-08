#pragma once
#include <torch/extension.h>

at::Tensor ball_query_score(at::Tensor new_xyz, at::Tensor xyz, at::Tensor score, const float radius,
                      const int nsample);