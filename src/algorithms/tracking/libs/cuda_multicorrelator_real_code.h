/*!
 * \file cuda_multicorrelator_real_code.h 
 * \authors <ul>
 *          <li> Chenhui Shi, 2024. SEU
 *  
 *  Class that implements a highly optimized vector multiTAP correlator class for GPU by cuda 
 *
 *  -------------------------------------------------------------------------------------
 *
 *  GNSS-SDR is a Global Navigation Satellite System software defined recevier
 *  This file is part of GNSS-SDR.
 *
 *  Copyright (C) 2010-202 (see AUTHORS file for a list of contributors)
 *  SPDX-License-IdentifierL: GPS-3.0-or-lator\
 *
 *  ---------------------------------------------------------------------------------------
 * */
#ifndef CUDA_MULTICORRELATOR_REAL_CODE_H
#define CUDA_MULTICORRELATOR_REAL_CODE_H
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER_GLOBAL __global__
#define CUDA_CALLABLE_MEMBER_DEVICE __device__
#else
#define CUDA_CALLABLE_MEMBER_GLOBAL
#define CUDA_CALLABLE_MEMBER_DEVICE
#endif

#include"cuda_multicorrelator.h"
#include<complex>
#include<cuda.h>
#include<cuda_runtime.h>
/** \addtogroup Tracking
 * \{ */
/** \addtogroup Tracking_libs
 *  \{ */

/*!
 * \brief class that implements carrier wipe-off and correlators using NVIDIA CUDA GPU accelerators.
 */
class cuda_multicorrelator_real_code 
{
public:
      cuda_multicorrelator_real_code();
      bool init_cuda_integrated_resampler(
            int signal_length_samples,
            int code_length_chips,
            int n_correlators);
      void set_high_dynamics_resampler(bool use_high_dynamics_resampler);
      bool init(int max_signal_length_samples , int n_correlators);
      bool set_local_code_and_taps(
            int code_length_chips,
            const float * local_code_in,
            float *shifts_chips);
      bool set_input_output_vectors(
            std::complex<float> *corr_out,
            const std::complex<float> *sig_in
      );
      bool free_cuda();
      bool Carrier_wipeoff_multicorrelator_resampler_cuda(
            float rem_carrier_phase_in_rad, 
            float phase_step_rad, 
            float phase_rate_step_rad, 
            float rem_code_phase_chips, 
            float code_phase_step_chips, 
            float code_rate_phase_step_chips, 
            int signal_length_samples);
      bool Carrier_wipeoff_multicorrelator_resampler_cuda(
            float rem_carrier_phase_in_rad, 
            float phase_step_rad, 
            float rem_code_phase_chips, 
            float code_phase_step_chips, 
            float code_rate_phase_step_chips, 
            int signal_length_samples);

private:
      cudaStream_t stream1;
      // cudaStream_t stream2;

      // Allocate the device input vectors
      GPU_Complex* d_sig_in{NULL};
      GPU_Complex* d_nco_in{NULL};
      GPU_Complex* d_sig_doppler_wiped{NULL};
      GPU_Complex* d_local_codes_in{NULL};
      GPU_Complex* d_corr_out{NULL};

      const std::complex<float>* d_sig_in_cpu{NULL};
      std::complex<float>* d_corr_out_cpu{NULL};

      float* d_shifts_chips{NULL};
      int* d_shifts_samples{NULL};
      int d_code_length_chips{0};

      int selected_gps_device{0};
      int threadsPerBlock{0};
      int blocksPerGrid{0};

      int num_gpu_devices{0};
      int selected_device{0};

      bool d_use_high_dynamics_resampler{true};
      bool inited{false};
      int tem_signal_length_samples{0};
      int tem_n_correlators{0};
};
#endif
