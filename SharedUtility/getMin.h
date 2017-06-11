/**
 * devUtility.h
 * @brief: This file contains InitCUDA() function and a reducer class CReducer
 * Created on: May 24, 2012
 * Author: Zeyi Wen
 * Copyright @DBGroup University of Melbourne
 **/

#ifndef GETMIN_H_
#define GETMIN_H_
#include <cuda_runtime.h>
#include "DataType.h"

__device__ void GetMinValueOriginal(real*, int*);
__device__ void GetMinValueOriginal(real*);

__device__  int getBlockMin(const float *values, int *index);

#endif /* GETMIN_H_ */
