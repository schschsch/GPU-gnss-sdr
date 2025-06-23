/*!
 * \file cuda_multicorrelator.cu
 * \brief Highly optimized CUDA GPU vector multiTAP correlator class
 * \authors <ul>
 *          <li> Javier Arribas, 2015. jarribas(at)cttc.es
 *          </ul>
 *
 * Class that implements a highly optimized vector multiTAP correlator class for NVIDIA CUDA GPUs
 *
 * -----------------------------------------------------------------------------
 *
 * Copyright (C) 2010-2020  (see AUTHORS file for a list of contributors)
 *
 * GNSS-SDR is a software defined Global Navigation
 *          Satellite Systems receiver
 *
 * This file is part of GNSS-SDR.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * -----------------------------------------------------------------------------
 */
#include"cuda_multicorrelator_real_code.h"
#include<iostream>
#include<stdio.h>
#include<cuda_runtime.h>

namespace{
  const int ACCUM_N = 128;
}
__global__ void float2complex(const float * in, GPU_Complex * out, const int& elementN){
  for(int index = gridDim.x * blockIdx.x + threadIdx.x ; index < elementN ; index += blockDim.x * gridDim.x){
    out[index] = GPU_Complex(0.0F,in[index]);
  }
  return;
}
__global__ void Doppler_wippe_scalarProdGPUCPXxN_shifts_chips(
  GPU_Complex *d_corr_out,
  GPU_Complex *d_sig_in,
  GPU_Complex *d_sig_wiped,
  GPU_Complex *d_local_codes_in,
  float *d_shifts_chips,
  int code_length_chips,
  float code_phase_step_chips,
  float code_rate_phase_step_chips,
  float rem_code_phase_chips,
  int vectorN,
  int elementN,
  float rem_carrier_phase_in_rad,
  float phase_step_rad,
  float phase_rate_step_rad,
  bool use_high_dynamics_resampler)
{
    __shared__ GPU_Complex accumResult[ACCUM_N];
    //CUDA version of floating point NCO and vector dot product integrated
    float sin,cos;
    if(use_high_dynamics_resampler)
    {
      for(int i=blockIdx.x * blockDim.x + threadIdx.x;
          i < elementN;
          i += blockDim.x * gridDim.x)
        {
            __sincosf(rem_carrier_phase_in_rad + i * phase_step_rad + i * i * phase_rate_step_rad , &sin , &cos);
            d_sig_wiped[i] = d_sig_in[i] * GPU_Complex( cos , -sin );
        }
    }
    else{
        for(int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < elementN;
            i += blockDim.x * gridDim.x)
        {
              __sincosf(rem_carrier_phase_in_rad + i * phase_step_rad , &sin , &cos);
              d_sig_wiped[i] = d_sig_in[i] * GPU_Complex( cos , - sin );
        }
    }

    __syncthreads();
    //compute each vector with one block
    for(int vec=blockIdx.x ; vec < vectorN ; vec += gridDim.x){
      //compute a vector with ACCUM_N iAccum ACCUM_N should be equal to blockDim
      for(int iAccum = threadIdx.x ; iAccum < ACCUM_N ; iAccum += blockDim.x){
        GPU_Complex sum(0,0);
        float local_code_chip_index=0.0;
        for(int pos = iAccum; pos < elementN; pos += ACCUM_N){
          
          if(use_high_dynamics_resampler)
          local_code_chip_index = fmodf(code_phase_step_chips * __int2float_rd(pos) + code_rate_phase_step_chips * pos * pos + d_shifts_chips[vec] - rem_code_phase_chips ,code_length_chips);
          else{local_code_chip_index = fmodf(code_phase_step_chips * __int2float_rd(pos) + d_shifts_chips[vec] - rem_code_phase_chips ,code_length_chips);}
          
          if(local_code_chip_index < 0) local_code_chip_index += code_length_chips - 1;
          
          local_code_chip_index = fmodf((local_code_chip_index+(d_shifts_chips[vec]-d_shifts_chips[0])/code_phase_step_chips), 
                                        elementN);
          
          sum.multiply_acc(d_sig_wiped[pos] , d_local_codes_in[__float2int_rd(local_code_chip_index)]);
        }
      accumResult[iAccum] = sum;
      }
      //compute sum of accumResult in log(ACCUM_N) times
      for (int stride = ACCUM_N / 2 ; stride > 0; stride >>= 1){
        __syncthreads();

        for(int iAccum=threadIdx.x;iAccum < stride; iAccum += blockDim.x){
          accumResult[iAccum] += accumResult[iAccum + stride];
        }
      }
      if(threadIdx.x ==0){
        d_corr_out[vec] = accumResult[0];
      }
    }
}

//version without phase_rate_step_rad
__global__ void Doppler_wippe_scalarProdGPUCPXxN_shifts_chips(
    GPU_Complex *d_corr_out,
    GPU_Complex *d_sig_in,
    GPU_Complex *d_sig_wiped,
    GPU_Complex *d_local_codes_in,
    float *d_shifts_chips,
    int code_length_chips,
    float code_phase_step_chips,
    float code_rate_phase_step_chips,
    float rem_code_phase_chips,
    int vectorN,
    int elementN,
    float rem_carrier_phase_in_rad,
    float phase_step_rad,
    bool use_high_dynamics_resampler
){
    __shared__ GPU_Complex iAccumResult[ACCUM_N]; 
    float sin,cos;
    //compute wiped sign with cuda function
    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
    i < elementN;
    i += blockDim.x * gridDim.x)
    {
      __sincosf(rem_carrier_phase_in_rad + i * phase_step_rad , &sin , &cos);
      d_sig_wiped[i] = d_sig_in[i] * GPU_Complex( cos , - sin );
    }
    __syncthreads();

    //compute each vector with one block
    for(int vec=blockIdx.x ; vec < vectorN ; vec += gridDim.x){
      //compute a vector with ACCUM_N iAccum ACCUM_N should be equal to blockDim
      for(int iAccum = threadIdx.x ; iAccum < ACCUM_N ; iAccum += blockDim.x){
        GPU_Complex sum(0,0);
        float local_code_chip_index=0.0;
        for(int pos = iAccum; pos < elementN; pos += ACCUM_N){
          if(use_high_dynamics_resampler)
          {
              local_code_chip_index = fmodf(code_phase_step_chips * __int2float_rd(pos) + code_rate_phase_step_chips * pos * pos + d_shifts_chips[0] - rem_code_phase_chips ,code_length_chips);
          }
          else
          {
              local_code_chip_index = fmodf(code_phase_step_chips * __int2float_rd(pos) + d_shifts_chips[vec] - rem_code_phase_chips ,code_length_chips);
          }
          if(local_code_chip_index < 0){
              local_code_chip_index += code_length_chips - 1;
          }
          if(use_high_dynamics_resampler)
          {
              local_code_chip_index =fmodf( ( local_code_chip_index + ( d_shifts_chips[vec] - d_shifts_chips[0] ) / code_phase_step_chips ),
                                            elementN) ;
          }
          sum.multiply_acc(d_sig_wiped[pos] , d_local_codes_in[__float2int_rd(local_code_chip_index)]);
        }
      iAccumResult[iAccum] = sum;
      }
      //compute sum of accumResult in log(ACCUM_N) times
      for (int stride = ACCUM_N / 2 ; stride > 0; stride >>= 1){
        __syncthreads();

        for(int iAccum=threadIdx.x;iAccum < stride; iAccum += blockDim.x){
          iAccumResult[iAccum] += iAccumResult[iAccum + stride];
        }
      }
      if(threadIdx.x ==0){
        d_corr_out[vec] = iAccumResult[0];
      }
    } 
}
bool cuda_multicorrelator_real_code::init_cuda_integrated_resampler(
  int signal_length_samples,
  int code_length_chips,
  int n_correlators)
{
    tem_n_correlators=n_correlators;
    tem_signal_length_samples=signal_length_samples;

    cudaDeviceProp prop;
    int num_devices, device;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1)
        {
            int max_multiprocessors = 0, max_device = 0;
            for (device = 0; device < num_devices; device++)
                {
                    cudaDeviceProp properties;
                    cudaGetDeviceProperties(&properties, device);
                    if (max_multiprocessors < properties.multiProcessorCount)
                        {
                            max_multiprocessors = properties.multiProcessorCount;
                            max_device = device;
                        }
                    printf("Found GPU device # %i\n", device);
                }
            selected_gps_device = rand() % num_devices;  //generates a random number between 0 and num_devices to split the threads between GPUs
            cudaSetDevice(selected_gps_device);

            cudaGetDeviceProperties(&prop, max_device);
            if (prop.canMapHostMemory != 1)
                {
                    printf("Device can not map memory.\n");
                }
            printf("L2 Cache size= %u \n", prop.l2CacheSize);
            printf("maxThreadsPerBlock= %u \n", prop.maxThreadsPerBlock);
            printf("maxGridSize= %i \n", prop.maxGridSize[0]);
            printf("sharedMemPerBlock= %lu \n", prop.sharedMemPerBlock);
            printf("deviceOverlap= %i \n", prop.deviceOverlap);
            printf("multiProcessorCount= %i \n", prop.multiProcessorCount);
        }
    else
        {
            cudaGetDevice(&selected_gps_device);
            cudaGetDeviceProperties(&prop, selected_gps_device);
            if (prop.canMapHostMemory != 1)
                {
                    printf("Device can not map memory.\n");
                }

            printf("L2 Cache size= %u \n", prop.l2CacheSize);
            printf("maxThreadsPerBlock= %u \n", prop.maxThreadsPerBlock);
            printf("maxGridSize= %i \n", prop.maxGridSize[0]);
            printf("sharedMemPerBlock= %lu \n", prop.sharedMemPerBlock);
            printf("deviceOverlap= %i \n", prop.deviceOverlap);
            printf("multiProcessorCount= %i \n", prop.multiProcessorCount);
        }
    size_t size = signal_length_samples * sizeof(GPU_Complex);
    cudaMalloc((void **)&d_sig_doppler_wiped, size);
    cudaMemset(d_sig_doppler_wiped, 0, size);
    cudaMalloc((void **)&d_local_codes_in, sizeof(std::complex<float>) * code_length_chips);
    cudaMemset(d_local_codes_in, 0, sizeof(std::complex<float>) * code_length_chips);

    d_code_length_chips = code_length_chips;

    cudaMalloc((void **)&d_shifts_chips, sizeof(float) * n_correlators);
    cudaMemset(d_shifts_chips, 0, sizeof(float) * n_correlators);


    threadsPerBlock = 64;
    blocksPerGrid = 128;  //(int)(signal_length_samples+threadsPerBlock-1)/threadsPerBlock;

    cudaStreamCreate(&stream1);
    inited=true;
    return true;
}

void cuda_multicorrelator_real_code::set_high_dynamics_resampler(
    bool use_high_dynamics_resampler)
{
  d_use_high_dynamics_resampler=use_high_dynamics_resampler;
}

bool cuda_multicorrelator_real_code::set_local_code_and_taps(
  int code_length_chips,
  const float * local_codes_in,
  float *shifts_chips)
{
    if(!inited){
      this->init_cuda_integrated_resampler(
        tem_signal_length_samples,
        code_length_chips,
        tem_n_correlators);
    }
    // std::complex<float> *local_codes_in_complex = new std::complex<float>[code_length_chips];
    // for (int i = 0 ; i < code_length_chips ; ++i) local_codes_in_complex[i] = std::complex<float>(0.0F , local_codes_in[i]);
    // cudaSetDevice(selected_gps_device);
    // cudaMemcpyAsync( d_local_codes_in, local_codes_in_complex, sizeof(GPU_Complex) * code_length_chips, 
    //   cudaMemcpyHostToDevice ,stream1);
    float *d_local_codes_in_float;
    cudaMalloc((void **)&d_local_codes_in_float, sizeof(float) * code_length_chips);
    cudaMemcpyAsync(d_local_codes_in_float, local_codes_in, sizeof(float) * code_length_chips, cudaMemcpyHostToDevice, stream1);
    float2complex<<<blocksPerGrid,threadsPerBlock,0,stream1>>>(d_local_codes_in_float , d_local_codes_in, code_length_chips);
    cudaFree(d_local_codes_in_float);
    d_code_length_chips = code_length_chips;

    cudaMemcpyAsync(d_shifts_chips, shifts_chips, sizeof(float) * tem_n_correlators,
      cudaMemcpyHostToDevice, stream1);

      return true;
}

bool cuda_multicorrelator_real_code::init(
    int max_signal_length_samples,
    int n_correlators)
{
    tem_signal_length_samples=max_signal_length_samples;
    tem_n_correlators=n_correlators;
    return true;
}

bool cuda_multicorrelator_real_code::set_input_output_vectors(
  std::complex<float> *corr_out ,
  const std::complex<float> *sig_in){
  cudaSetDevice(selected_gps_device);
  d_sig_in_cpu=sig_in;
  d_corr_out_cpu=corr_out;
  cudaError_t code;
  code = cudaHostGetDevicePointer((void **)&d_sig_in, (void *)sig_in, 0);
  code = cudaHostGetDevicePointer((void **)&d_corr_out, (void *)corr_out, 0);
  if(code !=cudaSuccess){
    printf("cuda cudaHostGetDevicePointer error \r\n");
  }
  return true;
}

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
        {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
}

bool cuda_multicorrelator_real_code::Carrier_wipeoff_multicorrelator_resampler_cuda(
    float rem_carrier_phase_in_rad,
    float phase_step_rad,
    float phase_rate_step_rad,
    float rem_code_phase_chips,
    float code_phase_step_chips,
    float code_rate_phase_step_chips,
    int signal_length_samples)
{
        cudaSetDevice(selected_gps_device);
        Doppler_wippe_scalarProdGPUCPXxN_shifts_chips<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
          d_corr_out,
          d_sig_in,
          d_sig_doppler_wiped,
          d_local_codes_in,
          d_shifts_chips,
          d_code_length_chips,
          code_phase_step_chips,
          code_rate_phase_step_chips,
          rem_code_phase_chips,
          tem_n_correlators,
          signal_length_samples,
          rem_carrier_phase_in_rad,
          phase_step_rad,
          phase_rate_step_rad,
          d_use_high_dynamics_resampler
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaStreamSynchronize(stream1));
        return true;
    }
bool cuda_multicorrelator_real_code::Carrier_wipeoff_multicorrelator_resampler_cuda(
    float rem_carrier_phase_in_rad, 
    float phase_step_rad, 
    float rem_code_phase_chips, 
    float code_phase_step_chips, 
    float code_rate_phase_step_chips, 
    int signal_length_samples)
{
        cudaSetDevice(selected_gps_device);
        Doppler_wippe_scalarProdGPUCPXxN_shifts_chips<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
          d_corr_out,
          d_sig_in,
          d_sig_doppler_wiped,
          d_local_codes_in,
          d_shifts_chips,
          d_code_length_chips,
          code_phase_step_chips,
          code_rate_phase_step_chips,
          rem_code_phase_chips,
          tem_n_correlators,
          signal_length_samples,
          rem_carrier_phase_in_rad,
          phase_step_rad,
          d_use_high_dynamics_resampler
        );
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaStreamSynchronize(stream1));
        return true;
}
cuda_multicorrelator_real_code::cuda_multicorrelator_real_code(){}
bool cuda_multicorrelator_real_code::free_cuda(){
  if (d_sig_in != NULL) cudaFree(d_sig_in);
  if (d_nco_in != NULL) cudaFree(d_nco_in);
  if (d_sig_doppler_wiped != NULL) cudaFree(d_sig_doppler_wiped);
  if (d_local_codes_in != NULL) cudaFree(d_local_codes_in);
  if (d_corr_out != NULL) cudaFree(d_corr_out);
  if (d_shifts_samples != NULL) cudaFree(d_shifts_samples);
  if (d_shifts_chips != NULL) cudaFree(d_shifts_chips);
  cudaDeviceReset();
  return true;
}
