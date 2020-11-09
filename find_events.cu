#define MACRO_THREADS 1024
#define NOT_EVENT 0

#define PPT 64
#define FILT_WINDOW 5
#define EVENTS 5

//1 = text file
//2 = binary file
#define READ_OPT 2

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <assert.h>
#include <string.h>

#include <fstream>
#include <iostream>

int SIZE; // length of the data file going in, in units of points...

#include "find_transitions_mean.h"
#include "find_transitions_delta.h"
#include "find_transitions_canny.h"
#include "find_high_random.h"
#include "mean_filter_signal.h"

float *h_values_raw;
float *h_values;
int *h_transitions;
float *h_high_mean;
float *d_values;
int *d_transitions;
float *d_high_mean;
int *d_size;
float *d_gradient;
float *d_smoothed;

int main(int argc, char ** argv) {

  int expected_values = EVENTS;
  int inBinaryFile;

  const char *available_kernels[4];
  available_kernels[0] = "mean";
  available_kernels[1] = "delta";
  available_kernels[2] = "canny";
  available_kernels[3] = "c";

  if (READ_OPT == 1)
    {
      printf("You are reading the CSV as an arg!\n");
      char *filename = strdup(argv[2]);
      std::ifstream inTextFile;
      // Open the file and read data
      inTextFile.open(filename);
      if (!inTextFile)
	{
	  printf("\nFailed to open file on csv run!\n");
	  return 1;
	}
      // Read SIZE from first line of the file
      inTextFile >> SIZE;
    }
  else
    {
      char filename[] = "signal.dat";
      if ((inBinaryFile = open(filename, O_RDONLY)) < 0)
	{
	  printf("\nFailed to open the file on dat run!\n");
	  return 1;
	}
      float *lines;
      lines = (float*)calloc(1, sizeof(float));
      read(inBinaryFile, lines, sizeof(float));
      SIZE = (*lines);
    }

  printf("%d was the number of points you just passed in.\n", SIZE);

  assert(SIZE > 0);

  const int THREADS = MACRO_THREADS;
  const int BLOCKS = floor((((float)SIZE / (float)THREADS)) / PPT);
  printf("Block count: %d.\n", BLOCKS);
  fflush(stdout);
  const int cropped_size = BLOCKS*THREADS*PPT;
  const int cropped_bytes = cropped_size*sizeof(float);
  printf("%d was as close as I could get.\n", cropped_size);
  fflush(stdout);

  assert(THREADS > 0);
  assert(BLOCKS > 0);
  assert(cropped_size > 0);
  assert(cropped_size < SIZE);

  // Now, copy the input and drop the last few points

  const int ARRAY_BYTES = SIZE * sizeof(float);
  h_values_raw = (float *)calloc(SIZE, sizeof(float));
  if (READ_OPT == 1)
    {
      char *filename = strdup(argv[2]);
      std::ifstream inTextFile;
      //Open the file and read data
      inTextFile.open(filename);
      if (!inTextFile)
	{
	  printf("\nFailed to open file");
	  return 1;
	}
      printf("You are pulling data from the CSV.\n");
      for (int i = 0; i < SIZE; i++)
	inTextFile >> h_values_raw[i];
    }
  else
    {
      read(inBinaryFile, h_values_raw, ARRAY_BYTES);
    }

  cudaMallocHost((void**) &h_values, cropped_bytes);
  FILE *f;  // Regardless of CPU or GPU, this is the file you're writing results to.

  int i;
  for(i=1; i<cropped_size; i++)
    {
        h_values[i] = h_values_raw[i];
    }

  free(h_values_raw);
  h_transitions = (int*)calloc(BLOCKS, sizeof(int));
  h_high_mean = (float*)calloc(1, sizeof(float));

  if (argc == 1)
    {
      printf("Run with one of the arguments: ");
      for (int i = 0; i < sizeof(available_kernels) / sizeof(available_kernels[0]); i++)
	printf("%s ", available_kernels[i]);
      printf("\n");
      return 1;
    }

  if (strcmp(argv[1], "c") == 0)
    {
      printf("Using CPU.\n");
      // Now you are not using the GPU at all, and are just on C on the CPU.
    }

  else
    { // You are in the GPU branch.
      // Allocate GPU memory
      printf("Using GPU.\n");
      cudaMalloc((void**) &d_values, cropped_bytes);
      cudaMalloc((void**) &d_smoothed, cropped_bytes);
      cudaMalloc((void**) &d_transitions, sizeof(int) * BLOCKS);
      cudaMalloc((void**) &d_high_mean, sizeof(float));
      cudaMalloc((void**) &d_size, sizeof(int));
      cudaMalloc((void**) &d_gradient, cropped_bytes);

      cudaStream_t stream1;
      cudaStreamCreate(&stream1);
      cudaMemcpyAsync(d_values, h_values, cropped_bytes, cudaMemcpyHostToDevice, stream1);
      printf("Host-to-device copy initiated.\n");
      fflush(stdout);
      // Launch the kernel

      printf("All pre-kernel launch stuff OK.\n");
      fflush(stdout);
      if (strcmp(argv[1], "delta") == 0)
	{
	  // Transfer the array to GPU
	  cudaMemcpy(d_size, &cropped_size, sizeof(int), cudaMemcpyHostToDevice);
	  find_transitions_delta <<< BLOCKS, THREADS, 0, stream1>>>(d_values, d_transitions, PPT, MACRO_THREADS);
	  // copy the result back to CPU
	  f = fopen("transitions_guessed_delta.csv", "w");
	  expected_values = EVENTS*2;
	  cudaMemcpy(h_values, d_values, cropped_bytes, cudaMemcpyDeviceToHost);
	}
      else if (strcmp(argv[1], "mean") == 0)
	{
	  *h_high_mean = find_high_random(h_values);
	  cudaMemcpy(d_high_mean, h_high_mean, sizeof(float), cudaMemcpyHostToDevice);
	  find_transitions_mean <<< BLOCKS, THREADS, 0, stream1>>>(d_values, d_transitions, PPT, MACRO_THREADS, d_high_mean);
	  f = fopen("transitions_guessed_mean.csv", "w");
	  cudaMemcpy(h_values, d_values, cropped_bytes, cudaMemcpyDeviceToHost);
	}
      else if (strcmp(argv[1], "canny") == 0)
	{
	  // Transfer the array to GPU
	  cudaMemcpy(d_size, &cropped_size, sizeof(int), cudaMemcpyHostToDevice);
	  mean_filter_signal<<< BLOCKS, THREADS,0, stream1>>>(d_values, PPT, FILT_WINDOW, d_size, d_smoothed);
	  find_transitions_canny<<< BLOCKS, THREADS,0, stream1 >>>(d_values, d_transitions, PPT, d_size, d_gradient);
	  f = fopen("transitions_guessed_canny.csv", "w");
	  expected_values = EVENTS*2;
	  cudaMemcpy(h_values, d_gradient, cropped_bytes, cudaMemcpyDeviceToHost);
	}
      else
	{
	  printf("Run with one of the arguments: ");
	  for (int i = 0; i < sizeof(available_kernels) / sizeof(available_kernels[0]); i++)
	    printf("%s ", available_kernels[i]);
	  printf("\n");
	  return 1;
	}
      // free GPU memory
      cudaFree(d_values);
      cudaStreamDestroy(stream1);
    }

  // Write the found transitions to a file
  for (int i = 0; i < cropped_size; i++)
    fprintf(f, "%f\n", h_values[i]);
  fclose(f);

  char eventFlag = 'F';
  int total_transitions = 0;
  for (int i = 0; i < cropped_size; i++){
    if (h_values[i] == NOT_EVENT && eventFlag == 'F'){
      continue; // you're not in an event, and you pass
	}
    else if (h_values[i] != NOT_EVENT && eventFlag == 'F'){
      eventFlag = 'T'; // walked into event
    }
    else if (h_values[i] != NOT_EVENT && eventFlag == 'T'){
      continue; // moving along event
    }
    else if (h_values[i] == NOT_EVENT && eventFlag == 'T'){
      total_transitions++;
      eventFlag = 'F';
    }
    else { return 1;}
  }
  printf("Computed with %s : ", argv[1]);
  printf("%d (%d expected for synthetically generated data.)\n", total_transitions, expected_values);
}
