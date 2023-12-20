/* lightweight-cuda-mpi-profiler is a simple CUDA-Aware MPI profiler
 * Copyright (C) 2021 Yiltan Hassan Temucin
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef MAIN_H
#define MAIN_H

#include "mpi.h"
#include <stdio.h>
#include "cuda_helpers.h"
#include "mpi_helpers.h"

typedef enum
{
  MPI_ALLREDUCE,
  MPI_INIT,
  MPI_BROADCAST,
  MPI_GATHER,
  MPI_FINALIZE
} mpi_call_type;

const int number_of_counted_calls = 5;

typedef struct
{
  mpi_call_type this_call;
  char *name[];
} count_call_properties;

// count_call_properties count_call_props[number_of_counted_calls] = {
//   {MPI_ALLREDUCE, "MPI_AllReduce"},
//   {MPI_INIT, "MPI_Init"},
//   {MPI_BROADCAST, "MPI_Bcast"},
//   {MPI_GATHER, "MPI_Gather"},
//   {MPI_FINALIZE, "MPI_Finalize"}
// };

// MPI functions we plan to profile
int MPI_Init(int *argc, char ***argv);
int MPI_Init_thread(int *argc, char ***argv, int required, int *provided);
int MPI_Finalize(void);

// Collectives
int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

// Metrics we will profile
int GPU_buffers_larger_than_4MB;
int count_calls[number_of_counted_calls];

// Functions to count metrics
static inline void init_metrics() {
  GPU_buffers_larger_than_4MB = 0;
  int i;
  for(i = 0; i < number_of_counted_calls; i++)
    count_calls[i] = 0;
}

static inline void MPI_profile_count_metrics(int callType){
  count_calls[callType]++;
}

static inline void count_metrics(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype) {

  // Logic to count GPU_buffers_larger_than_4MB metric;
  int buffer_size = get_MPI_message_size(datatype, count);
  int buffer_is_larger_than_4MB = buffer_size >= (4 * 1024 * 1024);

  int is_GPU_buffer = is_device_pointer(sendbuf) | is_device_pointer(recvbuf);

  if (buffer_is_larger_than_4MB && is_GPU_buffer) {
    GPU_buffers_larger_than_4MB++;
  }

  // Add other metrics
}

static inline void print_metrics() {
  printf("This Application has %d GPU buffers larger than 4MB\n",
         GPU_buffers_larger_than_4MB++);
  int i;
  for(i = 0; i < number_of_counted_calls; i++)
  {
    switch(i){  
      case MPI_ALLREDUCE:
        printf("MPI_ALLREDUCE");
        break;
      case MPI_INIT:
        printf("MPI_INIT");
        break;
      case MPI_BROADCAST:
        printf("MPI_BROADCAST");
        break;
      case MPI_GATHER:
        printf("MPI_GATHER");
        break;
      case MPI_FINALIZE:
        printf("MPI_FINALIZE");
        break;
      default:
        printf("ERROR");
        break;
    }
    printf(" was called %d times\n", count_calls[i]);
  }
}

#endif // MAIN_H
