% Purpose :  LSTM

clear all; close all; clc;

% Load configuration file
config_basic
  
% Step 1 : Read data
if gradCheckFlag
    generate_randdata
else
    readdata_rnn
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

