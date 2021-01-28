__global__ void mean_filter_signal(float *d_values,
				   int d_ppt,
				   int window_width,
				   int *d_size,
				   float *d_smoothed
				   )
{
  int idx = blockIdx.x * blockDim.x * d_ppt + threadIdx.x*d_ppt;
  for (int i = idx; i < idx + d_ppt; i++) {
    float sum = 0;
    for (int j = 0; j <= window_width / 2; j++) {
      if (i - j < 0)
	sum += d_values[i];
      else
	sum += d_values[i - j];
    }
    for (int j = 1; j <= window_width / 2; j++) {
      if (i + j >= *d_size){
	sum += d_values[i];}
      else
	sum += d_values[i + j];
    }
    sum /= window_width;
    d_smoothed[i] = sum;
  }
}
