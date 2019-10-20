__global__ void find_transitions_canny(float *d_v,
					   int *d_trans,
					   int d_ppt,
					   int *d_s,
					   float *d_grad
					   )
{  __shared__ int found[MACRO_THREADS];

  //Initialize the array
  for (int i = 0; i < blockDim.x; i++)
    found[i] = 0;

  float EVENT_THRESHOLD = 1;
  //Tunable -- edge cutoff
  
  int idx = blockIdx.x*blockDim.x*(d_ppt) + threadIdx.x*(d_ppt);
  
  for (int i = idx; i < idx + (d_ppt); i++){
    if (i == 0)
      d_grad[i] = (d_v[i + 1] - d_v[i]) / 2;
    else if (i >= *d_s - 1)
      d_grad[*d_s - 1] = (d_v[*d_s - 1] - d_v[*d_s - 2]) / 2;
    else
      {d_grad[i] = fabsf((d_v[i + 1] - d_v[i - 1]) / 2);
	if (d_grad[i] > EVENT_THRESHOLD)
	  {found[threadIdx.x] = 1;
	    d_grad[i] = d_v[i];
	  }
	else d_grad[i] = NOT_EVENT;
      }
  }
  
  __syncthreads();
  
  if (threadIdx.x == 0){
    int block_trans = 0;
    if (found[0] == 1)
      block_trans++;
    // Avoid double counting of the same transition
    for (int i = 1; i < blockDim.x; i++)
      if (found[i] == 1 && found[i - 1] == 0)
	block_trans++;
    d_trans[blockIdx.x] = block_trans;
    
    // Discard edges
    for (int i = 0; i < 50; i++)
      d_grad[i]  = 0;
    for (int i = gridDim.x * blockDim.x * (d_ppt) - 1; i < (*d_s); i++)
      d_grad[i] = 0;
  }
}
