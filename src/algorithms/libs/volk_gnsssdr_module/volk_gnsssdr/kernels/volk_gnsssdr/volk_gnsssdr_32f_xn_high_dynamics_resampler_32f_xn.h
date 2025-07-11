/*!
 * \file volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn.h
 * \brief VOLK_GNSSSDR kernel: Resamples 1 complex 32-bit float vectors using zero hold resample algorithm
 * and produces the delayed replicas by copying and rotating the resulting resampled signal.
 * \authors <ul>
 *          <li> Cillian O'Driscoll, 2017. cillian.odirscoll(at)gmail.com
 *          <li> Javier Arribas, 2018. javiarribas(at)gmail.com
 *          </ul>
 *
 * VOLK_GNSSSDR kernel that resamples N 32-bit float vectors using zero hold resample algorithm.
 * It is optimized to resample a single GNSS local code signal replica into 1 vector fractional-resampled and fractional-delayed
 * and produces the delayed replicas by copying and rotating the resulting resampled signal.
 * (i.e. it creates the Early, Prompt, and Late code replicas)
 *
 * -----------------------------------------------------------------------------
 *
 * GNSS-SDR is a Global Navigation Satellite System software-defined receiver.
 * This file is part of GNSS-SDR.
 *
 * Copyright (C) 2010-2020  (see AUTHORS file for a list of contributors)
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * -----------------------------------------------------------------------------
 */

/*!
 * \page volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn
 *
 * \b Overview
 *
 * Resamples a 32-bit floating point vector , providing \p num_out_vectors outputs.
 *
 * <b>Dispatcher Prototype</b>
 * \code
 * void volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn(float** result, const float* local_code, float rem_code_phase_chips, float code_phase_step_chips, float code_phase_rate_step_chips, float* shifts_chips, unsigned int code_length_chips, int num_out_vectors, unsigned int num_points)
 * \endcode
 *
 * \b Inputs
 * \li local_code:                 Vector to be resampled.
 * \li rem_code_phase_chips:       Remnant code phase [chips].
 * \li code_phase_step_chips:      Phase increment per sample [chips/sample].
 * \li code_phase_rate_step_chips: Phase rate increment per sample [chips/sample^2].
 * \li shifts_chips:               Vector of floats that defines the spacing (in chips) between the replicas of \p local_code
 * \li code_length_chips:          Code length in chips.
 * \li num_out_vectors             Number of output vectors.
 * \li num_points:                 The number of data values to be in the resampled vector.
 *
 * \b Outputs
 * \li result:                     Pointer to a vector of pointers where the results will be stored.
 *
 */

#ifndef INCLUDED_volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_H
#define INCLUDED_volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_H

#include <volk_gnsssdr/volk_gnsssdr_common.h>
#include <volk_gnsssdr/volk_gnsssdr_complex.h>
#include <assert.h>
#include <math.h>
#include <stdint.h> /* int64_t */
#include <stdio.h>
#include <stdlib.h> /* abs */


#ifdef LV_HAVE_GENERIC

static inline void volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_generic(
    float** result,
    const float* local_code, 
    float rem_code_phase_chips, 
    float code_phase_step_chips, 
    float code_phase_rate_step_chips, 
    float* shifts_chips, 
    unsigned int code_length_chips, 
    int num_out_vectors, 
    unsigned int num_points)
{
    int local_code_chip_index;
    int current_correlator_tap;
    unsigned int n;
    // first correlator
    for (n = 0; n < num_points; n++)
        {
            // resample code for first tap
            local_code_chip_index = (int)floor(code_phase_step_chips * (float)n + code_phase_rate_step_chips * (float)(n * n) + shifts_chips[0] - rem_code_phase_chips);
            // Take into account that in multitap correlators, the shifts can be negative!
            if (local_code_chip_index < 0) local_code_chip_index += (int)code_length_chips * (abs(local_code_chip_index) / code_length_chips + 1);
            local_code_chip_index = local_code_chip_index % code_length_chips;
            result[0][n] = local_code[local_code_chip_index];
        }

    // adjacent correlators
    unsigned int shift_samples = 0;
    for (current_correlator_tap = 1; current_correlator_tap < num_out_vectors; current_correlator_tap++)
        {
            shift_samples += (int)round((shifts_chips[current_correlator_tap] - shifts_chips[current_correlator_tap - 1]) / code_phase_step_chips);
            memcpy(&result[current_correlator_tap][0], &result[0][shift_samples], (num_points - shift_samples) * sizeof(float));
            memcpy(&result[current_correlator_tap][num_points - shift_samples], &result[0][0], shift_samples * sizeof(float));
        }
}

#endif /*LV_HAVE_GENERIC*/


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
static inline void volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_a_sse3(float** result, const float* local_code, float rem_code_phase_chips, float code_phase_step_chips, float code_phase_rate_step_chips, float* shifts_chips, unsigned int code_length_chips, int num_out_vectors, unsigned int num_points)
{
    float** _result = result;
    const unsigned int quarterPoints = num_points / 4;
    //    int current_correlator_tap;
    unsigned int n;
    unsigned int k;
    int current_correlator_tap;
    const __m128 ones = _mm_set1_ps(1.0f);
    const __m128 fours = _mm_set1_ps(4.0f);
    const __m128 rem_code_phase_chips_reg = _mm_set_ps1(rem_code_phase_chips);
    const __m128 code_phase_step_chips_reg = _mm_set_ps1(code_phase_step_chips);
    const __m128 code_phase_rate_step_chips_reg = _mm_set_ps1(code_phase_rate_step_chips);

    __VOLK_ATTR_ALIGNED(16)
    int local_code_chip_index[4];
    int local_code_chip_index_;
    const __m128i zeros = _mm_setzero_si128();
    const __m128 code_length_chips_reg_f = _mm_set_ps1((float)code_length_chips);
    const __m128i code_length_chips_reg_i = _mm_set1_epi32((int)code_length_chips);
    __m128i local_code_chip_index_reg, aux_i, negatives, i;
    __m128 aux, aux2, aux3, indexnn, shifts_chips_reg, fi, igx, j, c, cTrunc, base;
    __m128 indexn = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);

    shifts_chips_reg = _mm_set_ps1((float)shifts_chips[0]);
    aux2 = _mm_sub_ps(shifts_chips_reg, rem_code_phase_chips_reg);

    for (n = 0; n < quarterPoints; n++)
        {
            aux = _mm_mul_ps(code_phase_step_chips_reg, indexn);
            indexnn = _mm_mul_ps(indexn, indexn);
            aux3 = _mm_mul_ps(code_phase_rate_step_chips_reg, indexnn);
            aux = _mm_add_ps(aux, aux3);
            aux = _mm_add_ps(aux, aux2);
            // floor
            i = _mm_cvttps_epi32(aux);
            fi = _mm_cvtepi32_ps(i);
            igx = _mm_cmpgt_ps(fi, aux);
            j = _mm_and_ps(igx, ones);
            aux = _mm_sub_ps(fi, j);

            // Correct negative shift
            c = _mm_div_ps(aux, code_length_chips_reg_f);
            aux3 = _mm_add_ps(c, ones);
            i = _mm_cvttps_epi32(aux3);
            cTrunc = _mm_cvtepi32_ps(i);
            base = _mm_mul_ps(cTrunc, code_length_chips_reg_f);
            local_code_chip_index_reg = _mm_cvtps_epi32(_mm_sub_ps(aux, base));
            negatives = _mm_cmplt_epi32(local_code_chip_index_reg, zeros);
            aux_i = _mm_and_si128(code_length_chips_reg_i, negatives);
            local_code_chip_index_reg = _mm_add_epi32(local_code_chip_index_reg, aux_i);

            _mm_store_si128((__m128i*)local_code_chip_index, local_code_chip_index_reg);

            for (k = 0; k < 4; ++k)
                {
                    _result[0][n * 4 + k] = local_code[local_code_chip_index[k]];
                }
            indexn = _mm_add_ps(indexn, fours);
        }

    for (n = quarterPoints * 4; n < num_points; n++)
        {
            // resample code for first tap
            local_code_chip_index_ = (int)floor(code_phase_step_chips * (float)n + code_phase_rate_step_chips * (float)(n * n) + shifts_chips[0] - rem_code_phase_chips);
            // Take into account that in multitap correlators, the shifts can be negative!
            if (local_code_chip_index_ < 0) local_code_chip_index_ += (int)code_length_chips * (abs(local_code_chip_index_) / code_length_chips + 1);
            local_code_chip_index_ = local_code_chip_index_ % code_length_chips;
            _result[0][n] = local_code[local_code_chip_index_];
        }

    // adjacent correlators
    unsigned int shift_samples = 0;
    for (current_correlator_tap = 1; current_correlator_tap < num_out_vectors; current_correlator_tap++)
        {
            shift_samples += (int)round((shifts_chips[current_correlator_tap] - shifts_chips[current_correlator_tap - 1]) / code_phase_step_chips);
            memcpy(&_result[current_correlator_tap][0], &_result[0][shift_samples], (num_points - shift_samples) * sizeof(float));
            memcpy(&_result[current_correlator_tap][num_points - shift_samples], &_result[0][0], shift_samples * sizeof(float));
        }
}
#endif


#ifdef LV_HAVE_SSE3
#include <pmmintrin.h>
static inline void volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_u_sse3(float** result, const float* local_code, float rem_code_phase_chips, float code_phase_step_chips, float code_phase_rate_step_chips, float* shifts_chips, unsigned int code_length_chips, int num_out_vectors, unsigned int num_points)
{
    float** _result = result;
    const unsigned int quarterPoints = num_points / 4;
    //    int current_correlator_tap;
    unsigned int n;
    unsigned int k;
    int current_correlator_tap;
    const __m128 ones = _mm_set1_ps(1.0f);
    const __m128 fours = _mm_set1_ps(4.0f);
    const __m128 rem_code_phase_chips_reg = _mm_set_ps1(rem_code_phase_chips);
    const __m128 code_phase_step_chips_reg = _mm_set_ps1(code_phase_step_chips);
    const __m128 code_phase_rate_step_chips_reg = _mm_set_ps1(code_phase_rate_step_chips);

    __VOLK_ATTR_ALIGNED(16)
    int local_code_chip_index[4];
    int local_code_chip_index_;
    const __m128i zeros = _mm_setzero_si128();
    const __m128 code_length_chips_reg_f = _mm_set_ps1((float)code_length_chips);
    const __m128i code_length_chips_reg_i = _mm_set1_epi32((int)code_length_chips);
    __m128i local_code_chip_index_reg, aux_i, negatives, i;
    __m128 aux, aux2, aux3, indexnn, shifts_chips_reg, fi, igx, j, c, cTrunc, base;
    __m128 indexn = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);

    shifts_chips_reg = _mm_set_ps1((float)shifts_chips[0]);
    aux2 = _mm_sub_ps(shifts_chips_reg, rem_code_phase_chips_reg);

    for (n = 0; n < quarterPoints; n++)
        {
            aux = _mm_mul_ps(code_phase_step_chips_reg, indexn);
            indexnn = _mm_mul_ps(indexn, indexn);
            aux3 = _mm_mul_ps(code_phase_rate_step_chips_reg, indexnn);
            aux = _mm_add_ps(aux, aux3);
            aux = _mm_add_ps(aux, aux2);
            // floor
            i = _mm_cvttps_epi32(aux);
            fi = _mm_cvtepi32_ps(i);
            igx = _mm_cmpgt_ps(fi, aux);
            j = _mm_and_ps(igx, ones);
            aux = _mm_sub_ps(fi, j);

            // Correct negative shift
            c = _mm_div_ps(aux, code_length_chips_reg_f);
            aux3 = _mm_add_ps(c, ones);
            i = _mm_cvttps_epi32(aux3);
            cTrunc = _mm_cvtepi32_ps(i);
            base = _mm_mul_ps(cTrunc, code_length_chips_reg_f);
            local_code_chip_index_reg = _mm_cvtps_epi32(_mm_sub_ps(aux, base));
            negatives = _mm_cmplt_epi32(local_code_chip_index_reg, zeros);
            aux_i = _mm_and_si128(code_length_chips_reg_i, negatives);
            local_code_chip_index_reg = _mm_add_epi32(local_code_chip_index_reg, aux_i);

            _mm_store_si128((__m128i*)local_code_chip_index, local_code_chip_index_reg);

            for (k = 0; k < 4; ++k)
                {
                    _result[0][n * 4 + k] = local_code[local_code_chip_index[k]];
                }
            indexn = _mm_add_ps(indexn, fours);
        }

    for (n = quarterPoints * 4; n < num_points; n++)
        {
            // resample code for first tap
            local_code_chip_index_ = (int)floor(code_phase_step_chips * (float)n + code_phase_rate_step_chips * (float)(n * n) + shifts_chips[0] - rem_code_phase_chips);
            // Take into account that in multitap correlators, the shifts can be negative!
            if (local_code_chip_index_ < 0) local_code_chip_index_ += (int)code_length_chips * (abs(local_code_chip_index_) / code_length_chips + 1);
            local_code_chip_index_ = local_code_chip_index_ % code_length_chips;
            _result[0][n] = local_code[local_code_chip_index_];
        }

    // adjacent correlators
    unsigned int shift_samples = 0;
    for (current_correlator_tap = 1; current_correlator_tap < num_out_vectors; current_correlator_tap++)
        {
            shift_samples += (int)round((shifts_chips[current_correlator_tap] - shifts_chips[current_correlator_tap - 1]) / code_phase_step_chips);
            memcpy(&_result[current_correlator_tap][0], &_result[0][shift_samples], (num_points - shift_samples) * sizeof(float));
            memcpy(&_result[current_correlator_tap][num_points - shift_samples], &_result[0][0], shift_samples * sizeof(float));
        }
}

#endif


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
static inline void volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_a_sse4_1(float** result, const float* local_code, float rem_code_phase_chips, float code_phase_step_chips, float code_phase_rate_step_chips, float* shifts_chips, unsigned int code_length_chips, int num_out_vectors, unsigned int num_points)
{
    float** _result = result;
    const unsigned int quarterPoints = num_points / 4;
    //    int current_correlator_tap;
    unsigned int n;
    unsigned int k;
    int current_correlator_tap;
    const __m128 ones = _mm_set1_ps(1.0f);
    const __m128 fours = _mm_set1_ps(4.0f);
    const __m128 rem_code_phase_chips_reg = _mm_set_ps1(rem_code_phase_chips);
    const __m128 code_phase_step_chips_reg = _mm_set_ps1(code_phase_step_chips);
    const __m128 code_phase_rate_step_chips_reg = _mm_set_ps1(code_phase_rate_step_chips);

    __VOLK_ATTR_ALIGNED(16)
    int local_code_chip_index[4];
    int local_code_chip_index_;
    const __m128i zeros = _mm_setzero_si128();
    const __m128 code_length_chips_reg_f = _mm_set_ps1((float)code_length_chips);
    const __m128i code_length_chips_reg_i = _mm_set1_epi32((int)code_length_chips);
    __m128i local_code_chip_index_reg, aux_i, negatives, i;
    __m128 aux, aux2, aux3, indexnn, shifts_chips_reg, c, cTrunc, base;
    __m128 indexn = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);

    shifts_chips_reg = _mm_set_ps1((float)shifts_chips[0]);
    aux2 = _mm_sub_ps(shifts_chips_reg, rem_code_phase_chips_reg);

    for (n = 0; n < quarterPoints; n++)
        {
            aux = _mm_mul_ps(code_phase_step_chips_reg, indexn);
            indexnn = _mm_mul_ps(indexn, indexn);
            aux3 = _mm_mul_ps(code_phase_rate_step_chips_reg, indexnn);
            aux = _mm_add_ps(aux, aux3);
            aux = _mm_add_ps(aux, aux2);
            // floor
            aux = _mm_floor_ps(aux);

            // Correct negative shift
            c = _mm_div_ps(aux, code_length_chips_reg_f);
            aux3 = _mm_add_ps(c, ones);
            i = _mm_cvttps_epi32(aux3);
            cTrunc = _mm_cvtepi32_ps(i);
            base = _mm_mul_ps(cTrunc, code_length_chips_reg_f);
            local_code_chip_index_reg = _mm_cvtps_epi32(_mm_sub_ps(aux, base));
            negatives = _mm_cmplt_epi32(local_code_chip_index_reg, zeros);
            aux_i = _mm_and_si128(code_length_chips_reg_i, negatives);
            local_code_chip_index_reg = _mm_add_epi32(local_code_chip_index_reg, aux_i);

            _mm_store_si128((__m128i*)local_code_chip_index, local_code_chip_index_reg);

            for (k = 0; k < 4; ++k)
                {
                    _result[0][n * 4 + k] = local_code[local_code_chip_index[k]];
                }
            indexn = _mm_add_ps(indexn, fours);
        }

    for (n = quarterPoints * 4; n < num_points; n++)
        {
            // resample code for first tap
            local_code_chip_index_ = (int)floor(code_phase_step_chips * (float)n + code_phase_rate_step_chips * (float)(n * n) + shifts_chips[0] - rem_code_phase_chips);
            // Take into account that in multitap correlators, the shifts can be negative!
            if (local_code_chip_index_ < 0) local_code_chip_index_ += (int)code_length_chips * (abs(local_code_chip_index_) / code_length_chips + 1);
            local_code_chip_index_ = local_code_chip_index_ % code_length_chips;
            _result[0][n] = local_code[local_code_chip_index_];
        }

    // adjacent correlators
    unsigned int shift_samples = 0;
    for (current_correlator_tap = 1; current_correlator_tap < num_out_vectors; current_correlator_tap++)
        {
            shift_samples += (int)round((shifts_chips[current_correlator_tap] - shifts_chips[current_correlator_tap - 1]) / code_phase_step_chips);
            memcpy(&_result[current_correlator_tap][0], &_result[0][shift_samples], (num_points - shift_samples) * sizeof(float));
            memcpy(&_result[current_correlator_tap][num_points - shift_samples], &_result[0][0], shift_samples * sizeof(float));
        }
}

#endif


#ifdef LV_HAVE_SSE4_1
#include <smmintrin.h>
static inline void volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_u_sse4_1(float** result, const float* local_code, float rem_code_phase_chips, float code_phase_step_chips, float code_phase_rate_step_chips, float* shifts_chips, unsigned int code_length_chips, int num_out_vectors, unsigned int num_points)
{
    float** _result = result;
    const unsigned int quarterPoints = num_points / 4;
    //    int current_correlator_tap;
    unsigned int n;
    unsigned int k;
    int current_correlator_tap;
    const __m128 ones = _mm_set1_ps(1.0f);
    const __m128 fours = _mm_set1_ps(4.0f);
    const __m128 rem_code_phase_chips_reg = _mm_set_ps1(rem_code_phase_chips);
    const __m128 code_phase_step_chips_reg = _mm_set_ps1(code_phase_step_chips);
    const __m128 code_phase_rate_step_chips_reg = _mm_set_ps1(code_phase_rate_step_chips);

    __VOLK_ATTR_ALIGNED(16)
    int local_code_chip_index[4];
    int local_code_chip_index_;
    const __m128i zeros = _mm_setzero_si128();
    const __m128 code_length_chips_reg_f = _mm_set_ps1((float)code_length_chips);
    const __m128i code_length_chips_reg_i = _mm_set1_epi32((int)code_length_chips);
    __m128i local_code_chip_index_reg, aux_i, negatives, i;
    __m128 aux, aux2, aux3, indexnn, shifts_chips_reg, c, cTrunc, base;
    __m128 indexn = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);

    shifts_chips_reg = _mm_set_ps1((float)shifts_chips[0]);
    aux2 = _mm_sub_ps(shifts_chips_reg, rem_code_phase_chips_reg);

    for (n = 0; n < quarterPoints; n++)
        {
            aux = _mm_mul_ps(code_phase_step_chips_reg, indexn);
            indexnn = _mm_mul_ps(indexn, indexn);
            aux3 = _mm_mul_ps(code_phase_rate_step_chips_reg, indexnn);
            aux = _mm_add_ps(aux, aux3);
            aux = _mm_add_ps(aux, aux2);
            // floor
            aux = _mm_floor_ps(aux);

            // Correct negative shift
            c = _mm_div_ps(aux, code_length_chips_reg_f);
            aux3 = _mm_add_ps(c, ones);
            i = _mm_cvttps_epi32(aux3);
            cTrunc = _mm_cvtepi32_ps(i);
            base = _mm_mul_ps(cTrunc, code_length_chips_reg_f);
            local_code_chip_index_reg = _mm_cvtps_epi32(_mm_sub_ps(aux, base));
            negatives = _mm_cmplt_epi32(local_code_chip_index_reg, zeros);
            aux_i = _mm_and_si128(code_length_chips_reg_i, negatives);
            local_code_chip_index_reg = _mm_add_epi32(local_code_chip_index_reg, aux_i);

            _mm_store_si128((__m128i*)local_code_chip_index, local_code_chip_index_reg);

            for (k = 0; k < 4; ++k)
                {
                    _result[0][n * 4 + k] = local_code[local_code_chip_index[k]];
                }
            indexn = _mm_add_ps(indexn, fours);
        }

    for (n = quarterPoints * 4; n < num_points; n++)
        {
            // resample code for first tap
            local_code_chip_index_ = (int)floor(code_phase_step_chips * (float)n + code_phase_rate_step_chips * (float)(n * n) + shifts_chips[0] - rem_code_phase_chips);
            // Take into account that in multitap correlators, the shifts can be negative!
            if (local_code_chip_index_ < 0) local_code_chip_index_ += (int)code_length_chips * (abs(local_code_chip_index_) / code_length_chips + 1);
            local_code_chip_index_ = local_code_chip_index_ % code_length_chips;
            _result[0][n] = local_code[local_code_chip_index_];
        }

    // adjacent correlators
    unsigned int shift_samples = 0;
    for (current_correlator_tap = 1; current_correlator_tap < num_out_vectors; current_correlator_tap++)
        {
            shift_samples += (int)round((shifts_chips[current_correlator_tap] - shifts_chips[current_correlator_tap - 1]) / code_phase_step_chips);
            memcpy(&_result[current_correlator_tap][0], &_result[0][shift_samples], (num_points - shift_samples) * sizeof(float));
            memcpy(&_result[current_correlator_tap][num_points - shift_samples], &_result[0][0], shift_samples * sizeof(float));
        }
}

#endif


#ifdef LV_HAVE_AVX
#include <immintrin.h>
static inline void volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_a_avx(float** result, const float* local_code, float rem_code_phase_chips, float code_phase_step_chips, float code_phase_rate_step_chips, float* shifts_chips, unsigned int code_length_chips, int num_out_vectors, unsigned int num_points)
{
    float** _result = result;
    const unsigned int avx_iters = num_points / 8;
    int current_correlator_tap;
    unsigned int n;
    unsigned int k;
    const __m256 eights = _mm256_set1_ps(8.0f);
    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 rem_code_phase_chips_reg = _mm256_set1_ps(rem_code_phase_chips);
    const __m256 code_phase_step_chips_reg = _mm256_set1_ps(code_phase_step_chips);
    const __m256 code_phase_rate_step_chips_reg = _mm256_set1_ps(code_phase_rate_step_chips);

    __VOLK_ATTR_ALIGNED(32)
    int local_code_chip_index[8];
    int local_code_chip_index_;

    const __m256 zeros = _mm256_setzero_ps();
    const __m256 code_length_chips_reg_f = _mm256_set1_ps((float)code_length_chips);
    const __m256 n0 = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);

    __m256i local_code_chip_index_reg, i;
    __m256 aux, aux2, aux3, shifts_chips_reg, c, cTrunc, base, negatives, indexn, indexnn;

    shifts_chips_reg = _mm256_set1_ps((float)shifts_chips[0]);
    aux2 = _mm256_sub_ps(shifts_chips_reg, rem_code_phase_chips_reg);
    indexn = n0;
    for (n = 0; n < avx_iters; n++)
        {
            __VOLK_GNSSSDR_PREFETCH_LOCALITY(&_result[0][8 * n + 7], 1, 0);
            __VOLK_GNSSSDR_PREFETCH_LOCALITY(&local_code_chip_index[8], 1, 3);
            aux = _mm256_mul_ps(code_phase_step_chips_reg, indexn);
            indexnn = _mm256_mul_ps(indexn, indexn);
            aux3 = _mm256_mul_ps(code_phase_rate_step_chips_reg, indexnn);
            aux = _mm256_add_ps(aux, aux3);
            aux = _mm256_add_ps(aux, aux2);
            // floor
            aux = _mm256_floor_ps(aux);

            // Correct negative shift
            c = _mm256_div_ps(aux, code_length_chips_reg_f);
            aux3 = _mm256_add_ps(c, ones);
            i = _mm256_cvttps_epi32(aux3);
            cTrunc = _mm256_cvtepi32_ps(i);
            base = _mm256_mul_ps(cTrunc, code_length_chips_reg_f);
            local_code_chip_index_reg = _mm256_cvttps_epi32(_mm256_sub_ps(aux, base));

            c = _mm256_cvtepi32_ps(local_code_chip_index_reg);
            negatives = _mm256_cmp_ps(c, zeros, 0x01);
            aux3 = _mm256_and_ps(code_length_chips_reg_f, negatives);
            aux = _mm256_add_ps(c, aux3);
            local_code_chip_index_reg = _mm256_cvttps_epi32(aux);

            _mm256_store_si256((__m256i*)local_code_chip_index, local_code_chip_index_reg);
            for (k = 0; k < 8; ++k)
                {
                    _result[0][n * 8 + k] = local_code[local_code_chip_index[k]];
                }
            indexn = _mm256_add_ps(indexn, eights);
        }

    for (n = avx_iters * 8; n < num_points; n++)
        {
            // resample code for first tap
            local_code_chip_index_ = (int)floor(code_phase_step_chips * (float)n + code_phase_rate_step_chips * (float)(n * n) + shifts_chips[0] - rem_code_phase_chips);
            // Take into account that in multitap correlators, the shifts can be negative!
            if (local_code_chip_index_ < 0) local_code_chip_index_ += (int)code_length_chips * (abs(local_code_chip_index_) / code_length_chips + 1);
            local_code_chip_index_ = local_code_chip_index_ % code_length_chips;
            _result[0][n] = local_code[local_code_chip_index_];
        }

    // adjacent correlators
    unsigned int shift_samples = 0;
    for (current_correlator_tap = 1; current_correlator_tap < num_out_vectors; current_correlator_tap++)
        {
            shift_samples += (int)round((shifts_chips[current_correlator_tap] - shifts_chips[current_correlator_tap - 1]) / code_phase_step_chips);
            memcpy(&_result[current_correlator_tap][0], &_result[0][shift_samples], (num_points - shift_samples) * sizeof(float));
            memcpy(&_result[current_correlator_tap][num_points - shift_samples], &_result[0][0], shift_samples * sizeof(float));
        }
}

#endif


#ifdef LV_HAVE_AVX
#include <immintrin.h>
static inline void volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_u_avx(float** result, const float* local_code, float rem_code_phase_chips, float code_phase_step_chips, float code_phase_rate_step_chips, float* shifts_chips, unsigned int code_length_chips, int num_out_vectors, unsigned int num_points)
{
    float** _result = result;
    const unsigned int avx_iters = num_points / 8;
    int current_correlator_tap;
    unsigned int n;
    unsigned int k;
    const __m256 eights = _mm256_set1_ps(8.0f);
    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 rem_code_phase_chips_reg = _mm256_set1_ps(rem_code_phase_chips);
    const __m256 code_phase_step_chips_reg = _mm256_set1_ps(code_phase_step_chips);
    const __m256 code_phase_rate_step_chips_reg = _mm256_set1_ps(code_phase_rate_step_chips);

    __VOLK_ATTR_ALIGNED(32)
    int local_code_chip_index[8];
    int local_code_chip_index_;

    const __m256 zeros = _mm256_setzero_ps();
    const __m256 code_length_chips_reg_f = _mm256_set1_ps((float)code_length_chips);
    const __m256 n0 = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);

    __m256i local_code_chip_index_reg, i;
    __m256 aux, aux2, aux3, shifts_chips_reg, c, cTrunc, base, negatives, indexn, indexnn;

    shifts_chips_reg = _mm256_set1_ps((float)shifts_chips[0]);
    aux2 = _mm256_sub_ps(shifts_chips_reg, rem_code_phase_chips_reg);
    indexn = n0;
    for (n = 0; n < avx_iters; n++)
        {
            __VOLK_GNSSSDR_PREFETCH_LOCALITY(&_result[0][8 * n + 7], 1, 0);
            __VOLK_GNSSSDR_PREFETCH_LOCALITY(&local_code_chip_index[8], 1, 3);
            aux = _mm256_mul_ps(code_phase_step_chips_reg, indexn);
            indexnn = _mm256_mul_ps(indexn, indexn);
            aux3 = _mm256_mul_ps(code_phase_rate_step_chips_reg, indexnn);
            aux = _mm256_add_ps(aux, aux3);
            aux = _mm256_add_ps(aux, aux2);
            // floor
            aux = _mm256_floor_ps(aux);

            // Correct negative shift
            c = _mm256_div_ps(aux, code_length_chips_reg_f);
            aux3 = _mm256_add_ps(c, ones);
            i = _mm256_cvttps_epi32(aux3);
            cTrunc = _mm256_cvtepi32_ps(i);
            base = _mm256_mul_ps(cTrunc, code_length_chips_reg_f);
            local_code_chip_index_reg = _mm256_cvttps_epi32(_mm256_sub_ps(aux, base));

            c = _mm256_cvtepi32_ps(local_code_chip_index_reg);
            negatives = _mm256_cmp_ps(c, zeros, 0x01);
            aux3 = _mm256_and_ps(code_length_chips_reg_f, negatives);
            aux = _mm256_add_ps(c, aux3);
            local_code_chip_index_reg = _mm256_cvttps_epi32(aux);

            _mm256_store_si256((__m256i*)local_code_chip_index, local_code_chip_index_reg);
            for (k = 0; k < 8; ++k)
                {
                    _result[0][n * 8 + k] = local_code[local_code_chip_index[k]];
                }
            indexn = _mm256_add_ps(indexn, eights);
        }

    for (n = avx_iters * 8; n < num_points; n++)
        {
            // resample code for first tap
            local_code_chip_index_ = (int)floor(code_phase_step_chips * (float)n + code_phase_rate_step_chips * (float)(n * n) + shifts_chips[0] - rem_code_phase_chips);
            // Take into account that in multitap correlators, the shifts can be negative!
            if (local_code_chip_index_ < 0) local_code_chip_index_ += (int)code_length_chips * (abs(local_code_chip_index_) / code_length_chips + 1);
            local_code_chip_index_ = local_code_chip_index_ % code_length_chips;
            _result[0][n] = local_code[local_code_chip_index_];
        }

    // adjacent correlators
    unsigned int shift_samples = 0;
    for (current_correlator_tap = 1; current_correlator_tap < num_out_vectors; current_correlator_tap++)
        {
            shift_samples += (int)round((shifts_chips[current_correlator_tap] - shifts_chips[current_correlator_tap - 1]) / code_phase_step_chips);
            memcpy(&_result[current_correlator_tap][0], &_result[0][shift_samples], (num_points - shift_samples) * sizeof(float));
            memcpy(&_result[current_correlator_tap][num_points - shift_samples], &_result[0][0], shift_samples * sizeof(float));
        }
}

#endif
//
//
// #ifdef LV_HAVE_NEON
// #include <arm_neon.h>
//
// static inline void volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_neon(float** result, const float* local_code, float rem_code_phase_chips, float code_phase_step_chips, float* shifts_chips, unsigned int code_length_chips, int num_out_vectors, unsigned int num_points)
// {
//    float** _result = result;
//    const unsigned int neon_iters = num_points / 4;
//    int current_correlator_tap;
//    unsigned int n;
//    unsigned int k;
//    const int32x4_t ones = vdupq_n_s32(1);
//    const float32x4_t fours = vdupq_n_f32(4.0f);
//    const float32x4_t rem_code_phase_chips_reg = vdupq_n_f32(rem_code_phase_chips);
//    const float32x4_t code_phase_step_chips_reg = vdupq_n_f32(code_phase_step_chips);
//
//    __VOLK_ATTR_ALIGNED(16)
//    int32_t local_code_chip_index[4];
//    int32_t local_code_chip_index_;
//
//    const int32x4_t zeros = vdupq_n_s32(0);
//    const float32x4_t code_length_chips_reg_f = vdupq_n_f32((float)code_length_chips);
//    const int32x4_t code_length_chips_reg_i = vdupq_n_s32((int32_t)code_length_chips);
//    int32x4_t local_code_chip_index_reg, aux_i, negatives, i;
//    float32x4_t aux, aux2, shifts_chips_reg, fi, c, j, cTrunc, base, indexn, reciprocal;
//    __VOLK_ATTR_ALIGNED(16)
//    const float vec[4] = {0.0f, 1.0f, 2.0f, 3.0f};
//    uint32x4_t igx;
//    reciprocal = vrecpeq_f32(code_length_chips_reg_f);
//    reciprocal = vmulq_f32(vrecpsq_f32(code_length_chips_reg_f, reciprocal), reciprocal);
//    reciprocal = vmulq_f32(vrecpsq_f32(code_length_chips_reg_f, reciprocal), reciprocal);  // this refinement is required!
//    float32x4_t n0 = vld1q_f32((float*)vec);
//
//    for (current_correlator_tap = 0; current_correlator_tap < num_out_vectors; current_correlator_tap++)
//        {
//            shifts_chips_reg = vdupq_n_f32((float)shifts_chips[current_correlator_tap]);
//            aux2 = vsubq_f32(shifts_chips_reg, rem_code_phase_chips_reg);
//            indexn = n0;
//            for (n = 0; n < neon_iters; n++)
//                {
//                    __VOLK_GNSSSDR_PREFETCH_LOCALITY(&_result[current_correlator_tap][4 * n + 3], 1, 0);
//                    __VOLK_GNSSSDR_PREFETCH(&local_code_chip_index[4]);
//                    aux = vmulq_f32(code_phase_step_chips_reg, indexn);
//                    aux = vaddq_f32(aux, aux2);
//
//                    // floor
//                    i = vcvtq_s32_f32(aux);
//                    fi = vcvtq_f32_s32(i);
//                    igx = vcgtq_f32(fi, aux);
//                    j = vcvtq_f32_s32(vandq_s32(vreinterpretq_s32_u32(igx), ones));
//                    aux = vsubq_f32(fi, j);
//
//                    // fmod
//                    c = vmulq_f32(aux, reciprocal);
//                    i = vcvtq_s32_f32(c);
//                    cTrunc = vcvtq_f32_s32(i);
//                    base = vmulq_f32(cTrunc, code_length_chips_reg_f);
//                    aux = vsubq_f32(aux, base);
//                    local_code_chip_index_reg = vcvtq_s32_f32(aux);
//
//                    negatives = vreinterpretq_s32_u32(vcltq_s32(local_code_chip_index_reg, zeros));
//                    aux_i = vandq_s32(code_length_chips_reg_i, negatives);
//                    local_code_chip_index_reg = vaddq_s32(local_code_chip_index_reg, aux_i);
//
//                    vst1q_s32((int32_t*)local_code_chip_index, local_code_chip_index_reg);
//
//                    for (k = 0; k < 4; ++k)
//                        {
//                            _result[current_correlator_tap][n * 4 + k] = local_code[local_code_chip_index[k]];
//                        }
//                    indexn = vaddq_f32(indexn, fours);
//                }
//            for (n = neon_iters * 4; n < num_points; n++)
//                {
//                    __VOLK_GNSSSDR_PREFETCH_LOCALITY(&_result[current_correlator_tap][n], 1, 0);
//                    // resample code for current tap
//                    local_code_chip_index_ = (int)floor(code_phase_step_chips * (float)n + shifts_chips[current_correlator_tap] - rem_code_phase_chips);
//                    //Take into account that in multitap correlators, the shifts can be negative!
//                    if (local_code_chip_index_ < 0) local_code_chip_index_ += (int)code_length_chips * (abs(local_code_chip_index_) / code_length_chips + 1);
//                    local_code_chip_index_ = local_code_chip_index_ % code_length_chips;
//                    _result[current_correlator_tap][n] = local_code[local_code_chip_index_];
//                }
//        }
// }
//
// #endif

#endif /* INCLUDED_volk_gnsssdr_32f_xn_high_dynamics_resampler_32f_xn_H */
