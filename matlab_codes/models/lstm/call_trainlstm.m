% Purpose :  LSTM

clear all; close all; clc;
 
% Load configuration file
config_basic

% Step 1 : Read data
% readdata_rnn
generate_randdata
train_test_numbats = round(train_numbats/2);

% Step 2 : Set architecture
arch_name = strcat(arch_name,num2str(dout),ol_type);
arch_init
arch_name = strcat(arch_name,'_',model_name,'_',wtinit_meth);

switch sgd_type
    
    case 'sgdcm'
        
        for lr = lr_vec
            for mf = mf_vec
                
                arch_name = strcat(arch_name,'_lr',num2str(lr),'_mf',num2str(mf), ...
                    '_si',num2str(si),'_sr',num2str(ri),'_su',num2str(so),'_fbinit',num2str(fb_init),...
                    '_',wtinit_meth,'_',num2str(nwt));
                nwt = nwt + 1;
            
                % weight initialization
                wt_init                
                
                % train the model
                train_lstm

            end
        end
        
        
    case 'adadelta'
        
        
        for rho = rho_vec
            for eps_hp = eps_vec
                
                arch_name = strcat(arch_name,'_rho',num2str(rho),'_eps',num2str(eps_hp), ...
                    '_si',num2str(si),'_sr',num2str(sr),'_su',num2str(su),...
                    '_',wtinit_meth,'_',num2str(nwt));
                nwt = nwt + 1;
                
                % open error file
                fid = fopen(strcat(err_dir,'err_',arch_name,'.err'),'w');
                
                % weight initialization
                wt_init                
                
                % train the model
                train_lstm                
               
            end
        end
        
        
    case 'adam'
        
        for alpha = alpha_vec
            for beta1 = beta1_vec
                for beta2 = beta2_vec
                    
                    arch_name = strcat(arch_name,'_rho',num2str(alpha),'_beta1',num2str(beta1),'_beta2',num2str(beta2), ...
                        '_si',num2str(si),'_sr',num2str(sr),'_su',num2str(su),...
                        '_',wtinit_meth,'_',num2str(nwt));
                    nwt = nwt + 1;
                    
                    % weight initialization
                    wt_init
                    
                    % train the model
                    train_lstm                    
                    
                end
            end
        end
        
end

