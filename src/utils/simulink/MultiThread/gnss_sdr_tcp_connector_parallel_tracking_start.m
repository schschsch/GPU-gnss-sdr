% This MATLAB function builds and configures a Simulink model
% for interacting with the GNSS-SDR platform through a TCP
% communication. Parallel Computing version.
% \author David Pubill, 2012. dpubill(at)cttc.es
%
% ----------------------------------------------------------------------
%
% Copyright (C) 2010-2012  (see AUTHORS file for a list of contributors)
%
% GNSS-SDR is a software defined Global Navigation
%          Satellite Systems receiver
%
% This file is part of GNSS-SDR.
%
% SPDX-License-Identifier: GPL-3.0-or-later
%
% ----------------------------------------------------------------------
%/

function gnss_sdr_tcp_connector_parallel_tracking_start(num_channels, num_cores)

    %The parallel for (parfor) loop allows to build and run a Simulink
    %model in parallel mode, programming different threads
    parfor i = 0:num_cores-1;

        %Open and close the Simulink Library
        simulink('open');
        simulink('close');

        %User parameters
        host = '84.88.61.86'; %Remote IP address (GNSS-SDR computer IP)
        port = 2070;          %Remote port (GNSS-SDR computer port for Ch0)
        num_vars_rx = 9;      %Number of variables expected from GNSS-SDR
        num_vars_tx = 4;      %Number of variable to be transmitted to GNSS-SDR
        timeout = '40';       %Timeout [s]

        %Name of the tracking block, it must match the Simulink model name
        tracking_block_name = 'gnss_sdr_tcp_connector_tracking';

        % Layout coordinates for the gnss_sdr_tcp_connector_tracking blocks
        X0 = 20;
        X1 = 170;
        Y0 = 20;
        Y1 = 140;
        X_offset = 200;
        Y_offset = 160;

        %Calculate the size of the data received from GNSS-SDR
        %(float = 4 bytes each variable)
        datasize_RX = num_vars_rx*4;

        %Create a Simulink model
        model_name = ['gnss_sdr_tcp_connector_parallel_tracking_aux_', num2str(i)];
        new_system(model_name);
        open_system(model_name);

        %Set parameters to avoid warnings in the Command Window
        set_param(model_name,...
        'InheritedTsInSrcMsg', 'none');
        warning('off', 'Simulink:Commands:SetParamLinkChangeWarn');

        %Assign values to the variables used by Simulink in the base workspace
        assignin('base', 'Ti', 1e-3);
        assignin('base', 'f0', 1.57542e9);
        assignin('base', 'SFunSlope', 3.5);
        assignin('base', 'Tc', 4e-3/4092);
        assignin('base', 'T', 1e-3);
        assignin('base', 'B_PLL', 50);
        assignin('base', 'B_DLL', 2);

        %Calculate some variables to control the number of blocks that
        %should content each Simulink model in function of the number of
        %cores specified
        min_num_blocks_per_model = floor(num_channels/num_cores);
        id = rem(num_channels,num_cores);

        if(i<id)
            aux=1;
        else
            aux=0;
        end

        %Build the Simulink model for the core 'i'
        for m = 0:min_num_blocks_per_model+aux-1

            index = m + min_num_blocks_per_model*i + min(id,i);

            %Add and prepare an empty block to become the TCP connector block
            tcp_connector_block=[model_name, '/gnss_sdr_tcp_connector_tracking_', num2str(index)];

            add_block('simulink/Ports & Subsystems/Subsystem', tcp_connector_block);
            delete_line(tcp_connector_block, 'In1/1', 'Out1/1')

            tcp_connector_tracking_i_In1 = [model_name,'/gnss_sdr_tcp_connector_tracking_',num2str(index),'/In1'];
            tcp_connector_tracking_i_Out1 = [model_name,'/gnss_sdr_tcp_connector_tracking_',num2str(index),'/Out1'];

            delete_block(tcp_connector_tracking_i_In1);
            delete_block(tcp_connector_tracking_i_Out1);

            %Add to the TCP connector block the receiver, the tracking and the
            %transmitter blocks
            tcp_connector_tracking_rx_block = [model_name,'/gnss_sdr_tcp_connector_tracking_',num2str(index),'/gnss_sdr_tcp_connector_tracking_rx'];
            tcp_connector_tracking_block = [model_name,'/gnss_sdr_tcp_connector_tracking_',num2str(index),'/', tracking_block_name];
            tcp_connector_tracking_tx_block = [model_name,'/gnss_sdr_tcp_connector_tracking_',num2str(index),'/gnss_sdr_tcp_connector_tracking_tx'];

            add_block('simulink/User-Defined Functions/gnss_sdr/gnss_sdr_tcp_connector_tracking_rx',tcp_connector_tracking_rx_block);

            path_to_tracking_block = ['simulink/User-Defined Functions/gnss_sdr/', tracking_block_name];
            add_block(path_to_tracking_block, tcp_connector_tracking_block);

            add_block('simulink/User-Defined Functions/gnss_sdr/gnss_sdr_tcp_connector_tracking_tx',tcp_connector_tracking_tx_block);

            %Connect the receiver block to the tracking block
            for j=1:num_vars_rx;
                rx_out_ports =['gnss_sdr_tcp_connector_tracking_rx/', num2str(j)];
                tracking_in_ports =[tracking_block_name, '/', num2str(j)];

                add_line(tcp_connector_block, rx_out_ports, tracking_in_ports)
            end

            %Connect the tracking block to the transmitter block
            for k=1:num_vars_tx;
                tracking_out_ports =[tracking_block_name, '/', num2str(k)];
                tx_in_ports =['gnss_sdr_tcp_connector_tracking_tx/',num2str(k)];

                add_line(tcp_connector_block, tracking_out_ports, tx_in_ports)
            end

            %Add, place and connect two scopes in the TCP connector block
            name_scope_1 = [tcp_connector_block,'/Scope'];
            add_block('simulink/Sinks/Scope', name_scope_1, 'Position', [500 300 550 350]);
            set_param(name_scope_1, 'NumInputPorts', '4', 'LimitDataPoints', 'off');
            add_line(tcp_connector_block, 'gnss_sdr_tcp_connector_tracking_rx/9', 'Scope/1', 'autorouting','on')

            tracking_scope_port2 = [tracking_block_name,'/2'];
            add_line(tcp_connector_block, tracking_scope_port2, 'Scope/2', 'autorouting','on')
            tracking_scope_port3 = [tracking_block_name,'/3'];
            add_line(tcp_connector_block, tracking_scope_port3, 'Scope/3', 'autorouting','on')
            tracking_scope_port4 = [tracking_block_name,'/4'];
            add_line(tcp_connector_block, tracking_scope_port4, 'Scope/4', 'autorouting','on')

            name_scope_2 = [tcp_connector_block,'/EPL'];
            add_block('simulink/Sinks/Scope', name_scope_2, 'Position', [500 400 550 450]);
            set_param(name_scope_2, 'LimitDataPoints', 'off');
            tracking_scope2_port5 = [tracking_block_name,'/5'];
            add_line(tcp_connector_block, tracking_scope2_port5, 'EPL/1', 'autorouting','on')

            num_port = port+index;

            %Set the TCP receiver parameters
            tcp_receiver = [model_name,'/gnss_sdr_tcp_connector_tracking_',num2str(index),'/gnss_sdr_tcp_connector_tracking_rx/RX'];
            set_param(tcp_receiver, 'Port', num2str(num_port), 'Host', host, 'DataSize', num2str(datasize_RX), 'Timeout', timeout);

            %Set the TCP transmitter parameters
            tcp_transmitter = [model_name, '/gnss_sdr_tcp_connector_tracking_',num2str(index),'/gnss_sdr_tcp_connector_tracking_tx/TX'];
            set_param(tcp_transmitter, 'Port', num2str(num_port), 'Host', host,'Timeout', timeout);

            %New layout coordinates for each block
            X2 = X0 + floor(m/4)*X_offset;
            X3 = X1 + floor(m/4)*X_offset;
            Y2 = Y0 + (m-4*floor(m/4))*Y_offset;
            Y3 = Y1 + (m-4*floor(m/4))*Y_offset;

            %Place the block in the layout
            set_param(tcp_connector_block, 'Position', [X2 Y2 X3 Y3]);
        end

        %Set parameters to configure the model Solver
        set_param(model_name,...
            'SolverType', 'Fixed-step', 'Solver', 'FixedStepDiscrete',...
            'FixedStep', 'auto', 'StopTime', 'inf');

        %Save the model with a definitive name
        model_name_ready = ['gnss_sdr_tcp_connector_parallel_tracking_ready_', num2str(i)];
        save_system(model_name, model_name_ready);

        %Pause the thread 'i*5' seconds in function of the number of core.
        %This allows the system to establish the TCP connections in the
        %correct order
        if (aux == 0)
               pause(i*5);
        end

        %Run the Simulink model
        set_param(model_name_ready,'simulationcommand','start');

    end
end
