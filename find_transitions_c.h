// Get the standard deviation of the values here, over the entire array (out to length).
float standard_dev(float *values,
		   int length)
{
  float sum = 0.0, mean, SD = 0.0;

  for (int i = 0; i < length; ++i) {
    sum += values[i];
  }

  mean = sum / length;

  for (int i = 0; i < length; ++i)
    SD += pow(values[i] - mean, 2);

  return sqrt(SD / length);
}

// This is the plain-vanilla transition finder for just doing this in straight C on the CPU, for reference.
// Inputs pointers to the values and transitions arrays, mean of the file, and pass count.
// Makes transitions array a 1 for event points, and 0 for not-event points, based on multipass finder
// Will modify the values array

void find_transitions_c(float *values,
			float *trans,
			float mean,
			int passes,
			int length)
{
  // First, start by taking everything from values and storing it in transitions
  for(int i = 0; i < length; i++)
    {
      trans[i] = values[i];
    }

  // This is the threshold for how many standard deviations below the mean you'd like to be to be in an event.
  int thresh = 4;
  int thresh_back = 2;
  assert(thresh-thresh_back > 0);

  // Get some index variables for event start and end.
  int event_start;
  int event_end;
  int event_flag = 0; // Assume we are not in an event.

  // This is the first standard deviation we'll use.
  float dev = standard_dev(values, length);

  // For each pass except the last pass, run eventfinding and replace points with mean value.
  // For the last pass, call the event.
  for (int i = 0; i < passes; i++){

    // Calculate new standard deviation, use that if it's lower.

    float new_dev = standard_dev(values, length);
    if (new_dev < dev){
      dev = new_dev;
    }
    float cutoff_in = mean-(thresh*dev);
    float cutoff_out = mean-((thresh-thresh_back)*dev);
    // Now, walk over array, and call events.
    for(int j=0; j<length; j++)
      {
	if (event_flag == 0){ // You are not in an event and are looking to enter.
	  if (values[j] < cutoff_in){ // You are in an event now.
	    event_flag = 1; // The event flag went up.
	    event_start = j; // Record where the event started.
	  }
	  if (values[j] >= cutoff_in){ // You are not in an event
	    if (i == passes - 1){ // If on the last pass, perform replacement to zero out non-event areas.
	      values[j] = NOT_EVENT;
	    }
	  }
	}
	if (event_flag == 1){ // You are in an event, and are looking to see if you are leaving.
	  if (values[j] >= cutoff_out){ // You are not in an event anymore, since you moved up by two SDs
	    event_flag = 0;
	    event_end = j;
	    assert (event_start < event_end);
	    if (i < passes - 1){ // If not on the last pass, do the replacement.
	      for(int k = event_start; k < event_end; k++){
		values[k] = mean;
	      }
	    }
	  }
	}
      }
    for(int i = 0; i < length; i++) // Take the points in values that are not in events, and zero them out in transitions, preserving the original transitions points (not the ones that got changed by the multipass replacement.
      {
	if (values[i] == NOT_EVENT){trans[i] = NOT_EVENT;}}
  }
  printf("Pure C eventfinding finished OK.\n");
}
