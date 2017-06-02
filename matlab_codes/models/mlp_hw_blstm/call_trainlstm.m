% Purpose :  LSTM

% function [] = call_trainlstm(arch_name1, sgd_type, si, fb_init, btbf, gcth, rho, eps_hp)
clear all; close all; clc;

sgd_type = 'adam';
arch_name1 = '5N5N5N5N5N5N5N';
si = '0.1';
fb_init = '0';
btbf = '0';
gcth = '1';
rho = '0.01';
eps_hp = '0.9';

sgd_type = sgd_type;
arch_name1 = arch_name1;
si = str2num(si);
fb_init = str2num(fb_init);
btbf = str2num(btbf);
gcth = str2num(gcth);

rho_hp = str2num(rho);
eps_hp = str2num(eps_hp);


% Load configuration file
config_blz17

% Step 1 : Read data
if gradCheckFlag
    sl = 10;
    generate_randdata
else
    readdata_rnn_blz17
    tot_train_numbats = train_numbats;
end

% Step 2 : Set architecture
arch_setup_step

% Step 3 : Start training
switch sgd_type
    
    case 'sgdcm'
        
        for lr = lr_vec
            for mf = mf_vec
                
                begin_training
                
            end
        end
        
    case 'adadelta'
        
        for rho_hp = rho_vec
            for eps_hp = eps_vec
                
                begin_training
                
            end
        end
        
    case 'adam'
        
        for alpha = alpha_vec
            for beta1 = beta1_vec
                for beta2 = beta2_vec
                    
                    begin_training
                    
                end
            end
        end
        
end

