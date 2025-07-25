/*!
 * \file gps_l1_ca_dll_pll_tracking_gpu_cc.cc
 * \brief Implementation of a code DLL + carrier PLL tracking block GPU ACCELERATED
 * \author Javier Arribas, 2015. jarribas(at)cttc.es
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

#include "gps_l1_ca_dll_pll_tracking_gpu_cc.h"
#include "GPS_L1_CA.h"
#include "gnss_satellite.h"
#include "gnss_sdr_flags.h"
#include "gps_sdr_signal_replica.h"
#include "lock_detectors.h"
#include "tracking_discriminators.h"
#include <boost/lexical_cast.hpp>
#include <glog/logging.h>
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <cuda_profiler_api.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>


gps_l1_ca_dll_pll_tracking_gpu_cc_sptr
gps_l1_ca_dll_pll_make_tracking_gpu_cc(
    int64_t fs_in,
    uint32_t vector_length,
    bool dump,
    std::string dump_filename,
    float pll_bw_hz,
    float dll_bw_hz,
    float early_late_space_chips)
{
    return gps_l1_ca_dll_pll_tracking_gpu_cc_sptr(new Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc(
        fs_in, vector_length, dump, dump_filename, pll_bw_hz, dll_bw_hz, early_late_space_chips));
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::forecast(int noutput_items,
    gr_vector_int &ninput_items_required)
{
    if (noutput_items != 0)
        {
            ninput_items_required[0] = static_cast<int32_t>(d_vector_length) * 2;  // set the required available samples in each call
        }
}


Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc(
    int64_t fs_in,
    uint32_t vector_length,
    bool dump,
    std::string dump_filename,
    float pll_bw_hz,
    float dll_bw_hz,
    float early_late_space_chips) : gr::block("Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc", gr::io_signature::make(1, 1, sizeof(gr_complex)),
                                        gr::io_signature::make(1, 1, sizeof(Gnss_Synchro)))
{
    // Telemetry bit synchronization message port input
    this->message_port_register_in(pmt::mp("preamble_timestamp_s"));
    this->message_port_register_out(pmt::mp("events"));
    this->message_port_register_in(pmt::mp("telemetry_to_trk"));
    // initialize internal vars
    d_dump = dump;
    d_fs_in = fs_in;

    //default to 2048
    d_vector_length = vector_length;
    d_dump_filename = dump_filename;
    d_correlation_length_samples = static_cast<int32_t>(d_vector_length);

    // Initialize tracking  ==========================================
    d_code_loop_filter.set_DLL_BW(dll_bw_hz);//滤波器初始化
    d_carrier_loop_filter.set_params(10.0, pll_bw_hz, 2);

    // --- DLL variables -------------------------------------------------------
    d_early_late_spc_chips = early_late_space_chips;  // Define early-late offset (in chips)

    // Set GPU flags
    cudaSetDeviceFlags(cudaDeviceMapHost);
    // allocate host memory
    // pinned memory mode - use special function to get OS-pinned memory
    d_n_correlator_taps = 3;  // Early, Prompt, and Late
    // Get space for a vector with the C/A code replica sampled 1x/chip
    cudaHostAlloc(reinterpret_cast<void **>(&d_ca_code), (static_cast<int32_t>(GPS_L1_CA_CODE_LENGTH_CHIPS) * sizeof(gr_complex)), cudaHostAllocMapped || cudaHostAllocWriteCombined);
    // Get space for the resampled early / prompt / late local replicas
    cudaHostAlloc(reinterpret_cast<void **>(&d_local_code_shift_chips), d_n_correlator_taps * sizeof(float), cudaHostAllocMapped || cudaHostAllocWriteCombined);
    cudaHostAlloc(reinterpret_cast<void **>(&in_gpu), 2 * d_vector_length * sizeof(gr_complex), cudaHostAllocMapped || cudaHostAllocWriteCombined);
    // correlator outputs (scalar)
    cudaHostAlloc(reinterpret_cast<void **>(&d_correlator_outs), sizeof(gr_complex) * d_n_correlator_taps, cudaHostAllocMapped || cudaHostAllocWriteCombined);

    // Set TAPs delay values [chips]
    d_local_code_shift_chips[0] = -d_early_late_spc_chips;
    d_local_code_shift_chips[1] = 0.0;
    d_local_code_shift_chips[2] = d_early_late_spc_chips;

    // --- Perform initializations ------------------------------
    multicorrelator_gpu = new cuda_multicorrelator();
    // local code resampler on GPU
    multicorrelator_gpu->init_cuda_integrated_resampler(2 * d_vector_length, GPS_L1_CA_CODE_LENGTH_CHIPS, d_n_correlator_taps);
    multicorrelator_gpu->set_input_output_vectors(d_correlator_outs, in_gpu);

    // define initial code frequency basis of NCO
    d_code_freq_chips = GPS_L1_CA_CODE_RATE_CPS;
    // define residual code phase (in chips)
    d_rem_code_phase_samples = 0.0;
    // define residual carrier phase
    d_rem_carrier_phase_rad = 0.0;

    // sample synchronization
    d_sample_counter = 0ULL;
    // d_sample_counter_seconds = 0;
    d_acq_sample_stamp = 0;

    d_enable_tracking = false;
    d_pull_in = false;

    // CN0 estimation and lock detector buffers
    d_cn0_estimation_counter = 0;
    d_Prompt_buffer = std::vector<gr_complex>(FLAGS_cn0_samples);
    d_carrier_lock_test = 1;
    d_CN0_SNV_dB_Hz = 0;
    d_carrier_lock_fail_counter = 0;
    d_carrier_lock_threshold = FLAGS_carrier_lock_th;

    systemName["G"] = std::string("GPS");
    systemName["S"] = std::string("SBAS");
    //set GNURADIO BLOCK
#if GNURADIO_GREATER_THAN_38
    this->set_relative_rate(1, static_cast<uint64_t>(d_vector_length * 2));
#else
    this->set_relative_rate(1.0 / static_cast<double>(d_vector_length * 2));
#endif

    d_acquisition_gnss_synchro = 0;
    d_channel = 0;
    d_acq_code_phase_samples = 0.0;
    d_acq_carrier_doppler_hz = 0.0;
    d_carrier_doppler_hz = 0.0;
    d_acc_carrier_phase_cycles = 0.0;
    d_code_phase_samples = 0.0;

    d_pll_to_dll_assist_secs_Ti = 0.0;
    d_rem_code_phase_chips = 0.0;
    d_code_phase_step_chips = 0.0;
    d_carrier_phase_step_rad = 0.0;

    d_acc_carrier_phase_initialized = false;
    // set_min_output_buffer((int64_t)300);
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::start_tracking()
{
    /*
     *  correct the code phase according to the delay between acq and trk
     */
    d_acq_code_phase_samples = d_acquisition_gnss_synchro->Acq_delay_samples;
    d_acq_carrier_doppler_hz = d_acquisition_gnss_synchro->Acq_doppler_hz;
    d_acq_sample_stamp = d_acquisition_gnss_synchro->Acq_samplestamp_samples;

    const int64_t acq_trk_diff_samples = static_cast<int64_t>(d_sample_counter) - static_cast<int64_t>(d_acq_sample_stamp);  // -d_vector_length;
    DLOG(INFO) << "Number of samples between Acquisition and Tracking =" << acq_trk_diff_samples;
    const double acq_trk_diff_seconds = static_cast<double>(acq_trk_diff_samples) / static_cast<double>(d_fs_in);
    // doppler effect
    // Fd=(C/(C+Vr))*F
    const double radial_velocity = (GPS_L1_FREQ_HZ + d_acq_carrier_doppler_hz) / GPS_L1_FREQ_HZ;
    // new chip and prn sequence periods based on acq Doppler
    d_code_freq_chips = radial_velocity * GPS_L1_CA_CODE_RATE_CPS;
    //多普勒频移下每个采样周期的码相位步长
    d_code_phase_step_chips = static_cast<double>(d_code_freq_chips) / static_cast<double>(d_fs_in);
    //每个码片的时间
    const double T_chip_mod_seconds = 1 / d_code_freq_chips;
    //prn码的时间
    const double T_prn_mod_seconds = T_chip_mod_seconds * GPS_L1_CA_CODE_LENGTH_CHIPS;
    //PRN采样点数
    const double T_prn_mod_samples = T_prn_mod_seconds * static_cast<double>(d_fs_in);
    //PRN采样点数
    d_correlation_length_samples = round(T_prn_mod_samples);
    //PRN输出时的时间
    const double T_prn_true_seconds = GPS_L1_CA_CODE_LENGTH_CHIPS / GPS_L1_CA_CODE_RATE_CPS;
    //PRN码真实采样点数
    const double T_prn_true_samples = T_prn_true_seconds * static_cast<double>(d_fs_in);
    //真实采样与修正采样时间差
    const double T_prn_diff_seconds = T_prn_true_seconds - T_prn_mod_seconds;
    //
    const double N_prn_diff = acq_trk_diff_seconds / T_prn_true_seconds;
    double corrected_acq_phase_samples, delay_correction_samples;
    corrected_acq_phase_samples = fmod((d_acq_code_phase_samples + T_prn_diff_seconds * N_prn_diff * static_cast<double>(d_fs_in)), T_prn_true_samples);
    if (corrected_acq_phase_samples < 0)
        {
            corrected_acq_phase_samples = T_prn_mod_samples + corrected_acq_phase_samples;
        }
    delay_correction_samples = d_acq_code_phase_samples - corrected_acq_phase_samples;

    d_acq_code_phase_samples = corrected_acq_phase_samples;

    d_carrier_doppler_hz = d_acq_carrier_doppler_hz;

    d_carrier_phase_step_rad = TWO_PI * d_carrier_doppler_hz / static_cast<double>(d_fs_in);

    // DLL/PLL filter initialization
    d_carrier_loop_filter.initialize(d_acq_carrier_doppler_hz);  // The carrier loop filter implements the Doppler accumulator
    d_code_loop_filter.initialize();                             // initialize the code filter

    // generate local reference ALWAYS starting at chip 1 (1 sample per chip)
    gps_l1_ca_code_gen_complex(own::span<gr_complex>(d_ca_code, static_cast<int32_t>(GPS_L1_CA_CODE_LENGTH_CHIPS)), d_acquisition_gnss_synchro->PRN, 0);

    multicorrelator_gpu->set_local_code_and_taps(static_cast<int32_t>(GPS_L1_CA_CODE_LENGTH_CHIPS), d_ca_code, d_local_code_shift_chips, d_n_correlator_taps);

    for (int32_t n = 0; n < d_n_correlator_taps; n++)
        {
            d_correlator_outs[n] = gr_complex(0, 0);
        }

    d_carrier_lock_fail_counter = 0;
    d_rem_code_phase_samples = 0.0;
    d_rem_carrier_phase_rad = 0.0;
    d_rem_code_phase_chips = 0.0;
    d_acc_carrier_phase_cycles = 0.0;
    d_pll_to_dll_assist_secs_Ti = 0.0;
    d_code_phase_samples = d_acq_code_phase_samples;

    sys = std::string(1, d_acquisition_gnss_synchro->System);

    // DEBUG OUTPUT
    std::cout << "Tracking of GPS L1 C/A signal started on channel " << d_channel << " for satellite " << Gnss_Satellite(systemName[sys], d_acquisition_gnss_synchro->PRN) << '\n';
    LOG(INFO) << "Tracking of GPS L1 C/A signal for satellite " << Gnss_Satellite(systemName[sys], d_acquisition_gnss_synchro->PRN) << " on channel " << d_channel;

    // enable tracking
    d_pull_in = true;
    d_enable_tracking = true;
    d_acc_carrier_phase_initialized = false;

    LOG(INFO) << "PULL-IN Doppler [Hz]=" << d_carrier_doppler_hz
              << " Code Phase correction [samples]=" << delay_correction_samples
              << " PULL-IN Code Phase [samples]=" << d_acq_code_phase_samples;
}


Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::~Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc()
{
    if (d_dump_file.is_open())
        {
            try
                {
                    d_dump_file.close();
                }
            catch (const std::exception &ex)
                {
                    LOG(WARNING) << "Exception in destructor " << ex.what();
                }
        }
    try
        {
            cudaFreeHost(in_gpu);
            cudaFreeHost(d_correlator_outs);
            cudaFreeHost(d_local_code_shift_chips);
            cudaFreeHost(d_ca_code);
            multicorrelator_gpu->free_cuda();
            delete (multicorrelator_gpu);
        }
    catch (const std::exception &ex)
        {
            LOG(WARNING) << "Exception in destructor " << ex.what();
        }
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::set_channel(uint32_t channel)
{
    d_channel = channel;
    LOG(INFO) << "Tracking Channel set to " << d_channel;
    // ############# ENABLE DATA FILE LOG #################
    if (d_dump == true)
        {
            if (d_dump_file.is_open() == false)
                {
                    try
                        {
                            d_dump_filename.append(boost::lexical_cast<std::string>(d_channel));
                            d_dump_filename.append(".dat");
                            d_dump_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
                            d_dump_file.open(d_dump_filename.c_str(), std::ios::out | std::ios::binary);
                            LOG(INFO) << "Tracking dump enabled on channel " << d_channel << " Log file: " << d_dump_filename.c_str();
                        }
                    catch (const std::ofstream::failure *e)
                        {
                            LOG(WARNING) << "channel " << d_channel << " Exception opening trk dump file " << e->what();
                        }
                }
        }
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::set_gnss_synchro(Gnss_Synchro *p_gnss_synchro)
{
    d_acquisition_gnss_synchro = p_gnss_synchro;
}


void Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::check_carrier_phase_coherent_initialization()
{
    if (d_acc_carrier_phase_initialized == false)
        {
            d_acc_carrier_phase_cycles = -d_rem_carrier_phase_rad / TWO_PI;
            d_acc_carrier_phase_initialized = true;
        }
}


int Gps_L1_Ca_Dll_Pll_Tracking_GPU_cc::general_work(int noutput_items __attribute__((unused)), gr_vector_int &ninput_items __attribute__((unused)),
    gr_vector_const_void_star &input_items, gr_vector_void_star &output_items)
{
    // Block input data and block output stream pointers
    const gr_complex *in = reinterpret_cast<const gr_complex *>(input_items[0]);
    Gnss_Synchro **out = reinterpret_cast<Gnss_Synchro **>(&output_items[0]);

    // GNSS_SYNCHRO OBJECT to interchange data between tracking->telemetry_decoder
    Gnss_Synchro current_synchro_data = Gnss_Synchro();

    // process vars
    double code_error_chips_Ti = 0.0;
    double code_error_filt_chips = 0.0;
    double code_error_filt_secs_Ti = 0.0;
    double CURRENT_INTEGRATION_TIME_S = 0.001;
    double CORRECTED_INTEGRATION_TIME_S = 0.001;
    double dll_code_error_secs_Ti = 0.0;
    double carr_phase_error_secs_Ti = 0.0;
    bool loss_of_lock = false;

    if (d_enable_tracking == true)
        {
            // Fill the acquisition data
            current_synchro_data = *d_acquisition_gnss_synchro;
            // Receiver signal alignment
            if (d_pull_in == true)
                {
                    const int32_t acq_to_trk_delay_samples = d_sample_counter - d_acq_sample_stamp;
                    const double acq_trk_shif_correction_samples = d_correlation_length_samples - fmod(static_cast<double>(acq_to_trk_delay_samples), static_cast<double>(d_correlation_length_samples));
                    const int32_t samples_offset = round(d_acq_code_phase_samples + acq_trk_shif_correction_samples);
                    current_synchro_data.Tracking_sample_counter = d_sample_counter + static_cast<uint64_t>(samples_offset);
                    current_synchro_data.fs = d_fs_in;
                    current_synchro_data.correlation_length_ms = 1;
                    *out[0] = std::move(current_synchro_data);
                    d_sample_counter += static_cast<uint64_t>(samples_offset);  // count for the processed samples
                    d_pull_in = false;
                    consume_each(samples_offset);  // shift input to perform alignment with local replica
                    return 1;
                }

            // ################# CARRIER WIPEOFF AND CORRELATORS ##############################
            // perform carrier wipe-off and compute Early, Prompt and Late correlation

            std::copy(in, in + d_correlation_length_samples, in_gpu);
            cudaProfilerStart();
            multicorrelator_gpu->Carrier_wipeoff_multicorrelator_resampler_cuda(static_cast<float>(d_rem_carrier_phase_rad),//0
                static_cast<float>(d_carrier_phase_step_rad),//0
                static_cast<float>(d_code_phase_step_chips),
                static_cast<float>(d_rem_code_phase_chips),
                d_correlation_length_samples, d_n_correlator_taps);
            cudaProfilerStop();
            // std::cout<<"c_out[0]="<<d_correlator_outs[0]<<"c_out[1]="<<d_correlator_outs[1]<<"c_out[2]="<<d_correlator_outs[2]<< '\n';

            // UPDATE INTEGRATION TIME
            CURRENT_INTEGRATION_TIME_S = static_cast<double>(d_correlation_length_samples) / static_cast<double>(d_fs_in);

            // ################## PLL ##########################################################
            // Update PLL discriminator [rads/Ti -> Secs/Ti]
            carr_phase_error_secs_Ti = pll_cloop_two_quadrant_atan(d_correlator_outs[1]) / TWO_PI;  // prompt output
            // Carrier discriminator filter
            // NOTICE: The carrier loop filter includes the Carrier Doppler accumulator, as described in Kaplan
            // d_carrier_doppler_hz = d_acq_carrier_doppler_hz + carr_phase_error_filt_secs_ti/INTEGRATION_TIME;
            // Input [s/Ti] -> output [Hz]
            d_carrier_doppler_hz = d_carrier_loop_filter.get_carrier_error(0.0, carr_phase_error_secs_Ti, CURRENT_INTEGRATION_TIME_S);
            // PLL to DLL assistance [Secs/Ti]
            d_pll_to_dll_assist_secs_Ti = (d_carrier_doppler_hz * CURRENT_INTEGRATION_TIME_S) / GPS_L1_FREQ_HZ;
            // code Doppler frequency update
            d_code_freq_chips = GPS_L1_CA_CODE_RATE_CPS + ((d_carrier_doppler_hz * GPS_L1_CA_CODE_RATE_CPS) / GPS_L1_FREQ_HZ);

            // ################## DLL ##########################################################
            // DLL discriminator
            code_error_chips_Ti = dll_nc_e_minus_l_normalized(d_correlator_outs[0], d_correlator_outs[2], d_early_late_spc_chips, 1.0);  // [chips/Ti]  // early and late
            // Code discriminator filter
            code_error_filt_chips = d_code_loop_filter.get_code_nco(code_error_chips_Ti);                      // input [chips/Ti] -> output [chips/second]
            code_error_filt_secs_Ti = code_error_filt_chips * CURRENT_INTEGRATION_TIME_S / d_code_freq_chips;  // [s/Ti]
            // DLL code error estimation [s/Ti]
            // TODO: PLL carrier aid to DLL is disabled. Re-enable it and measure performance
            dll_code_error_secs_Ti = -code_error_filt_secs_Ti + d_pll_to_dll_assist_secs_Ti;

            // ################## CARRIER AND CODE NCO BUFFER ALIGNMENT #######################
            // keep alignment parameters for the next input buffer
            // Compute the next buffer length based in the new period of the PRN sequence and the code phase error estimation
            const double T_chip_seconds = 1 / d_code_freq_chips;
            const double T_prn_seconds = T_chip_seconds * GPS_L1_CA_CODE_LENGTH_CHIPS;
            const double T_prn_samples = T_prn_seconds * static_cast<double>(d_fs_in);
            const double K_blk_samples = T_prn_samples + d_rem_code_phase_samples - dll_code_error_secs_Ti * static_cast<double>(d_fs_in);

            d_correlation_length_samples = round(K_blk_samples);                                           // round to a discrete samples
            d_rem_code_phase_samples = K_blk_samples - static_cast<double>(d_correlation_length_samples);  // rounding error < 1 sample

            // UPDATE REMNANT CARRIER PHASE
            CORRECTED_INTEGRATION_TIME_S = (static_cast<double>(d_correlation_length_samples) / static_cast<double>(d_fs_in));
            // remnant carrier phase [rad]
            d_rem_carrier_phase_rad = fmod(d_rem_carrier_phase_rad + TWO_PI * d_carrier_doppler_hz * CORRECTED_INTEGRATION_TIME_S, TWO_PI);
            // UPDATE CARRIER PHASE ACCUULATOR
            // carrier phase accumulator prior to update the PLL estimators (accumulated carrier in this loop depends on the old estimations!)
            d_acc_carrier_phase_cycles -= d_carrier_doppler_hz * CORRECTED_INTEGRATION_TIME_S;

            // ################### PLL COMMANDS #################################################
            // carrier phase step (NCO phase increment per sample) [rads/sample]
            d_carrier_phase_step_rad = TWO_PI * d_carrier_doppler_hz / static_cast<double>(d_fs_in);

            // ################### DLL COMMANDS #################################################
            // code phase step (Code resampler phase increment per sample) [chips/sample]
            d_code_phase_step_chips = d_code_freq_chips / static_cast<double>(d_fs_in);
            // remnant code phase [chips]
            d_rem_code_phase_chips = d_rem_code_phase_samples * (d_code_freq_chips / static_cast<double>(d_fs_in));

            // ####### CN0 ESTIMATION AND LOCK DETECTORS #######################################
            if (d_cn0_estimation_counter < FLAGS_cn0_samples)
                {
                    // fill buffer with prompt correlator output values
                    d_Prompt_buffer[d_cn0_estimation_counter] = d_correlator_outs[1];  // prompt
                    d_cn0_estimation_counter++;
                }
            else
                {
                    d_cn0_estimation_counter = 0;
                    // Code lock indicator
                    d_CN0_SNV_dB_Hz = cn0_m2m4_estimator(d_Prompt_buffer.data(), FLAGS_cn0_samples, GPS_L1_CA_CODE_PERIOD_S);
                    // Carrier lock indicator
                    d_carrier_lock_test = carrier_lock_detector(d_Prompt_buffer.data(), FLAGS_cn0_samples);
                    // Loss of lock detection
                    if (d_carrier_lock_test < d_carrier_lock_threshold or d_CN0_SNV_dB_Hz < FLAGS_cn0_min)
                        {
                            d_carrier_lock_fail_counter++;
                        }
                    else
                        {
                            if (d_carrier_lock_fail_counter > 0) d_carrier_lock_fail_counter--;
                        }
                    if (d_carrier_lock_fail_counter > FLAGS_max_lock_fail)
                        {
                            std::cout << "Loss of lock in channel " << d_channel << "!\n";
                            LOG(INFO) << "Loss of lock in channel " << d_channel << "!";
                            this->message_port_pub(pmt::mp("events"), pmt::from_long(3));  // 3 -> loss of lock
                            d_carrier_lock_fail_counter = 0;
                            d_enable_tracking = false;  // TODO: check if disabling tracking is consistent with the channel state machine
                            loss_of_lock = true;
                        }
                    check_carrier_phase_coherent_initialization();
                }

            // ########### Output the tracking data to navigation and PVT ##########
            current_synchro_data.Prompt_I = static_cast<double>((d_correlator_outs[1]).real());
            current_synchro_data.Prompt_Q = static_cast<double>((d_correlator_outs[1]).imag());
            current_synchro_data.Tracking_sample_counter = d_sample_counter + static_cast<uint64_t>(d_correlation_length_samples);
            current_synchro_data.Code_phase_samples = d_rem_code_phase_samples;
            current_synchro_data.Carrier_phase_rads = TWO_PI * d_acc_carrier_phase_cycles;
            current_synchro_data.Carrier_Doppler_hz = d_carrier_doppler_hz;
            current_synchro_data.CN0_dB_hz = d_CN0_SNV_dB_Hz;
            current_synchro_data.Flag_valid_symbol_output = !loss_of_lock;
            current_synchro_data.correlation_length_ms = 1;
        }
    else
        {
            for (int32_t n = 0; n < d_n_correlator_taps; n++)
                {
                    d_correlator_outs[n] = gr_complex(0, 0);
                }

            current_synchro_data.System = {'G'};
            current_synchro_data.correlation_length_ms = 1;
            current_synchro_data.Tracking_sample_counter = d_sample_counter + static_cast<uint64_t>(d_correlation_length_samples);
        }

    // assign the GNU Radio block output data
    current_synchro_data.fs = d_fs_in;
    *out[0] = std::move(current_synchro_data);

    if (d_dump)
        {
            // MULTIPLEXED FILE RECORDING - Record results to file
            float prompt_I;
            float prompt_Q;
            float tmp_E, tmp_P, tmp_L;
            float tmp_VE = 0.0;
            float tmp_VL = 0.0;
            float tmp_float;
            prompt_I = d_correlator_outs[1].real();
            prompt_Q = d_correlator_outs[1].imag();
            tmp_E = std::abs<float>(d_correlator_outs[0]);
            tmp_P = std::abs<float>(d_correlator_outs[1]);
            tmp_L = std::abs<float>(d_correlator_outs[2]);
            try
                {
                    // Dump correlators output
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_VE), sizeof(float));
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_E), sizeof(float));
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_P), sizeof(float));
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_L), sizeof(float));
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_VL), sizeof(float));
                    // PROMPT I and Q (to analyze navigation symbols)
                    d_dump_file.write(reinterpret_cast<char *>(&prompt_I), sizeof(float));
                    d_dump_file.write(reinterpret_cast<char *>(&prompt_Q), sizeof(float));
                    // PRN start sample stamp
                    d_dump_file.write(reinterpret_cast<char *>(&d_sample_counter), sizeof(uint64_t));
                    // accumulated carrier phase
                    tmp_float = static_cast<float>(d_acc_carrier_phase_cycles * TWO_PI);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    // carrier and code frequency
                    tmp_float = static_cast<float>(d_carrier_doppler_hz);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    tmp_float = static_cast<float>(d_code_freq_chips);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    // PLL commands
                    tmp_float = 1.0 / (carr_phase_error_secs_Ti * CURRENT_INTEGRATION_TIME_S);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    tmp_float = 1.0 / (code_error_filt_secs_Ti * CURRENT_INTEGRATION_TIME_S);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    // DLL commands
                    tmp_float = code_error_chips_Ti * CURRENT_INTEGRATION_TIME_S;
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    tmp_float = code_error_filt_secs_Ti;
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    // CN0 and carrier lock test
                    tmp_float = static_cast<float>(d_CN0_SNV_dB_Hz);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    tmp_float = static_cast<float>(d_carrier_lock_test);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    // AUX vars (for debug purposes)
                    tmp_float = code_error_chips_Ti * CURRENT_INTEGRATION_TIME_S;
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_float), sizeof(float));
                    double tmp_double = static_cast<double>(d_sample_counter + d_correlation_length_samples);
                    d_dump_file.write(reinterpret_cast<char *>(&tmp_double), sizeof(double));
                    // PRN
                    uint32_t prn_ = d_acquisition_gnss_synchro->PRN;
                    d_dump_file.write(reinterpret_cast<char *>(&prn_), sizeof(uint32_t));
                }
            catch (const std::ofstream::failure *e)
                {
                    LOG(WARNING) << "Exception writing trk dump file " << e->what();
                }
        }

    consume_each(d_correlation_length_samples);        // this is necessary in gr::block derivates
    d_sample_counter += d_correlation_length_samples;  // count for the processed samples

    if (d_enable_tracking || loss_of_lock)
        {
            return 1;
        }
    else
        {
            return 0;
        }
}
