__global__ void find_transitions_delta(float *d_values,
					   int *d_trans,
					   int d_ppt,
					   int thread_count)
{
  int idx = blockIdx.x * blockDim.x * d_ppt+threadIdx.x * d_ppt;
  float max = -10000;
  float min = 10000;

  assert(d_values[0] > max);
  assert(d_values[0] < min);
  for (int i = idx; i < idx + (d_ppt); i++)
    {
      if (d_values[i] > max){
	max = d_values[i];
      }
      if (d_values[i] < min){
	min = d_values[i];
      }
    }
  assert(max >= min);

  const float delta_threshold = 2;
  // Tunable -- units close to std

  if (abs(max - min) > delta_threshold)
    {
      if (d_ppt > 2)
	{
	  int recurse_d_ppt = d_ppt/2; // Both of these are always a power of 2...
	  int recurse_threads = 2;
	  int recurse_blocks = 1;
	  float *recurse_d_values = d_values+idx;
	  int *recurse_d_trans = d_trans+idx;
	  find_transitions_delta<<<recurse_blocks, recurse_threads>>>(recurse_d_values, recurse_d_trans, recurse_d_ppt, recurse_threads);
	}
      else
	{
	  assert(d_ppt <= 2);
	  for (int i = idx; i < idx + d_ppt; i++)
	    d_values[i] = d_values[i];
	}
    }
  else {
    assert((abs(max - min) < delta_threshold));
    for (int i = idx; i < idx + d_ppt; i++)
      d_values[i] = NOT_EVENT;
  }
}
