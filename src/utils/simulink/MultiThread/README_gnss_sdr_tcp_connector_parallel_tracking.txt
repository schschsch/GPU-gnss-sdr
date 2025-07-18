/*!
 * \file README.txt
 * \brief How to add a block to the Simulink Library repository of Matlab,
 * how to use the "gnss_sdr_tcp_connector_parallel_tracking_start.m" script
 * and how to replace the tracking block of the library. Parallel Computing
 * version.
 *
 * \author David Pubill, 2012. dpubill(at)cttc.es
 *
 * -------------------------------------------------------------------------
 *
 * Copyright (C) 2010-2012  (see AUTHORS file for a list of contributors)
 *
 * GNSS-SDR is a software defined Global Navigation
 *          Satellite Systems receiver
 *
 * This file is part of GNSS-SDR.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * -------------------------------------------------------------------------
 */


IMPORTANT: Please, to use this tracking check the configuration file called
'gnss-sdr_tcp_connector_tracking.conf'. There are two major changes:
	1.- Choose the [GPS_L1_CA_TCP_CONNECTOR_Tracking] tracking algorithm.
	2.- Choose a tcp port for channel 0 (e.g. Tracking.port_ch0=2070;)


A) HOW TO add a block to the Simulink Library repository of your Matlab installation
   ---------------------------------------------------------------------------------
 (These steps should be followed only the first time)

1.- Copy the content of this folder to a folder accessible from Simulink.

2.- In the Matlab Command Window type:
	>> simulink;
    to open the Simulink Library Browser.

3.- Right-click on the Simulink/User-Defined Functions of the Simulink
    Library menu, and click on "Open User-Defined Functions library"
    (Window_1).

4.- Open the library model 'gnss_sdr_tcp_connector_tracking_lib.mdl'
    (Window_2)

5.- If this is not the first time there should be an existing 'gnss-sdr'
    block in the 'User-Defined Functions' window that should be deleted
    before drag and drop the new 'gnss_sdr' block (which includes 3 blocks:
        - 'gnss_sdr_tcp_connector_tracking_rx' block
        - 'gnss_sdr_tcp_connector_tracking' block
        - 'gnss_sdr_tcp_connector_tracking_tx' block)
    from Window_2 to Window_1. A new message should appear: "This library
    is locked. The action performed requires it to be unlocked". Then,
    click on the "Unlock" button (the block will be copied) and close
    Window_2.

6.- Right-click on the 'gnss-sdr' block and click on "Link Options -->
    Disable link", repeat the action but now clicking on "Link Options -->
    Break link". This action disables and breaks the link with the
    original library model.

7.- On Window_1 save the "simulink/User-Defined Functions" library.
    To do that go to "File > Save". Then, close Window_1.

8.- From "Simulink Library Browser" window, press F5 to refresh and generate
    the new Simulink Library repository (it may take a few seconds). This
    completes the installation of the custom Simulink block.


B) HOW TO use the "gnss_sdr_tcp_connector_parallel_tracking_start.m" script:
   ----------------------------------------------------------------

-----------------------     ------------------     -----------------------
|                     |     |                |     |                     |
| gnss_sdr_tcp_       |     | gnss_sdr_tcp_  |     | gnss_sdr_tcp_       |
| connector_tracking_ | --> | connector_     | --> | connector_tracking_ |
| rx                  |     | tracking       |     | tx                  |
|                     |     |                |     |                     |
-----------------------     ------------------     -----------------------

The 'gnss_sdr_tcp_connector_parallel_tracking_start.m' is the script that
builds and configures a Simulink model for interacting with the GNSS-SDR
platform through a TCP communication. Some 'User parameters' can be
modified but, by default, these are the values assigned:

%User parameters
    host = '84.88.61.86'; %Remote IP address (GNSS-SDR computer IP)
    port = 2070;          %Remote port (GNSS-SDR computer port for Ch0)
    num_vars_rx = 9;      %Number of variables expected from GNSS-SDR
    num_vars_tx = 4;      %Number of variable to be transmitted to GNSS-SDR
    timeout = '40';       %Timeout in seconds

'host', 'port' and 'timeout' parameters configure both
'gnss_sdr_tcp_connector_tracking_rx' and 'gnss_sdr_tcp_connector_tracking_tx'
blocks. The 'port' parameter sets the base port number for the first
channel (ch0). Each of the subsequent channels increases their port by one
unit (e.g. ch0_port=2070, ch1_port=2071,...).

Also the name of the tracking block can be modified. It must match with
the Simulink model name:

    %Name of the tracking block, it must match the Simulink model name
    tracking_block_name = 'gnss_sdr_tcp_connector_tracking';

To configure the MATLAB to work in parallel mode (the 'Parallel Computing'
Toolbox must be installed in the MATLAB) type in the Matlab Command Window
the following:

>> matlabpool(C)

where C is the number of cores of the computer to be used.

Then it should appear a message like this one:

"Destroying 1 pre-existing parallel job(s) created by matlabpool that were
in the finished or failed state.

Starting matlabpool using the 'local' configuration ... connected to 4
labs."

Once the MATLAB is configured to work in parallel mode, type the following
to run the script:

>> gnss_sdr_tcp_connector_parallel_tracking_start(N,C);

where N must match the number of channels configured in the GNSS-SDR
platform and C is the same as before.

Note: to stop working with the parallel mode type in the Command Window
the following:

>> matlabpool close


C) HOW TO replace the tracking block of the library
   ------------------------------------------------

1.- Open the library model 'gnss_sdr_tcp_connector_tracking_lib.mdl'
2.- Unlock the library. Click on "Edit > Unlock Library".
3.- Open the "gnss-sdr" block and change the "gnss_sdr_tcp_connector_tracking"
    block by another one. If the name is different it must be updated in
    the "gnss_sdr_tcp_connector_parallel_tracking_start.m" code (see
    section B)
4.- Save the new library.
5.- Go to section A and follow the instructions.
