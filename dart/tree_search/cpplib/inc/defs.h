#pragma once

#include <cstddef>
#include <cstdint>

#include "omp.h"

#define NUM_THREADS 16

typedef int32_t token_t;
typedef int32_t freq_t;
typedef float prob_t;