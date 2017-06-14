% Purpose :  LSTM

clear all; close all; clc;

enc_arch_name = '4N4N4N4N4N4N4N';
dec_arch_name = '8N4N4N4N4N4N';
arch_name1 = strcat(enc_arch_name,dec_arch_name);


% Load configuration file
config_basic

% % Step 1 : Read data
% if gradCheckFlag
%     generate_randdata
% else
%     readdata_rnn
% end
din = 6;
dout = 3;

train_batchdata = 0; train_batchtargets=0; train_clv_s= 0; train_clv_t=0;
train_numbats = 6999;
val_numbats = 253;
test_numbats = 253;

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

