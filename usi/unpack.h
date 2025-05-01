#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cppshogi.h"

void unpack_features1(const int batch_size, packed_features1_t* p1, features1_t* x1, cudaStream_t stream);
