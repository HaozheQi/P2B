#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_score_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        const float *__restrict__ score,
                                        float *__restrict__ unique_score) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  score += batch_index * n;
  unique_score += m * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float s = score[k];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        unique_score[j] += s;
        ++cnt;
      }
    }
  }
}

void query_ball_point_score_kernel_wrapper(int b, int n, int m, float radius,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const float *score, float *unique_score) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_score_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_xyz, xyz, score, unique_score);

  CUDA_CHECK_ERRORS();
}
