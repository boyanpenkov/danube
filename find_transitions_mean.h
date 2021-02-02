__global__ void find_transitions_mean(float *d_values, int *d_trans, int d_ppt,
                                      int thread_count, float *d_high) {
  int idx = blockIdx.x * blockDim.x * d_ppt + threadIdx.x * d_ppt;
  float mean = 0;

  const float thresh = 0.9;
  // Tunable -- std of transition.

  if (d_ppt > 2) {
    for (int i = idx; i < idx + (d_ppt); i++)
      mean += d_values[i];
    mean /= (float)(d_ppt);
    float dev = fabsf(mean - *d_high);

    if (dev > thresh && mean < *d_high) {
      // There may be a transition here, but be careful, you're wide..
      int recurse_d_ppt = d_ppt / 2; // Both of these are always a power of 2...
      int recurse_threads = MACRO_THREADS;
      int recurse_blocks = floor((((float)d_ppt / (float)recurse_threads)) /
                                 (float)recurse_d_ppt);
      float *recurse_d_values = d_values + idx;
      int *recurse_d_trans = d_trans + idx;
      find_transitions_mean<<<recurse_blocks, recurse_threads>>>(
          recurse_d_values, recurse_d_trans, recurse_d_ppt, recurse_threads,
          d_high);
    } else {
      for (int i = idx; i < idx + d_ppt; i++)
        d_values[i] = NOT_EVENT;
    }
  } else {
    for (int i = idx; i < idx + d_ppt; i++)
      mean += d_values[i];
    mean /= (float)(d_ppt);

    float dev = fabsf(mean - *d_high);

    if (dev > thresh && mean < *d_high) {
      for (int i = idx; i < idx + d_ppt; i++)
        d_values[i] = mean;
    } else {
      for (int i = idx; i < idx + d_ppt; i++)
        d_values[i] = NOT_EVENT;
    }
  }
}
