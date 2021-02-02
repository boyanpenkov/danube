float find_high_random(float *values) {
  int interval_size = INTERVAL;
  int trials = 5;
  // Find 'trials' different mean values, and pick the highest
  float mean_high[trials];

  for (int i = 0; i < trials; i++) {
    mean_high[i] = 0;
  }

  int start_index = 0;

  for (int i = 0; i < trials; i++) {
    for (int j = start_index; j < start_index + interval_size; j++)
      mean_high[i] += values[j];

    mean_high[i] /= interval_size;
    // generate a random sampling index
    start_index = rand() % ((SIZE - interval_size) + 1);
  }

  float max = -100000;

  for (int i = 0; i < trials; i++)
    if (mean_high[i] > max)
      max = mean_high[i];

  return max;
}
