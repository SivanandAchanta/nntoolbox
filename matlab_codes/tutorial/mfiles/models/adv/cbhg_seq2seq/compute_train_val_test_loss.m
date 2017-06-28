% Check Validation Loss
if mod(num_up,check_valfreq) == 0
    
    tic
    [train_err] = compute_error(train_batchdata,train_batchtargets,train_clv_s,train_clv_t,train_test_numbats,p_f_1_N1,p_f_2_N1,p_c_1_N1,p_c_2_N1,p_c_3_N1,p_h_1_N1,p_h_2_N1,p_h_3_N1,p_h_4_N1,p_lf_1_N1,p_lb_1_N1,p_f_3_0,p_f_3_1,p_f_3_2,p_lf_1_dec,... 
p_lf_2_dec,p_lf_3_dec,p_f_4_1_dec,p_f_4_2_dec,p_f_1,p_f_2,f,nl,cfn,dp,K_conv_l1_n1,C_conv_l1_n1,K_conv_l2_n1,C_conv_l2_n1,enc_layers,Em);
    toc
    
    tic
    [val_err] = compute_error(val_batchdata,val_batchtargets,val_clv_s,val_clv_t,val_numbats,p_f_1_N1,p_f_2_N1,p_c_1_N1,p_c_2_N1,p_c_3_N1,p_h_1_N1,p_h_2_N1,p_h_3_N1,p_h_4_N1,p_lf_1_N1,p_lb_1_N1,p_f_3_0,p_f_3_1,p_f_3_2,p_lf_1_dec,p_lf_2_dec,p_lf_3_dec,p_f_4_1_dec,p_f_4_2_dec,p_f_1,p_f_2,f,nl,cfn,dp,K_conv_l1_n1,C_conv_l1_n1,K_conv_l2_n1,C_conv_l2_n1,enc_layers,Em);
    toc
    
    
    fprintf('Epoch: %d, Train Loss: %f; Val Loss : %f \n',NE,train_err,val_err);
    
    if val_err < best_val_err
        
        best_val_err = val_err;
        best_epoch = NE;

        test_err = val_err;

        %tic
        %[test_err] = compute_error(test_batchdata,test_batchtargets,test_clv_s,test_clv_t,test_numbats,p_f_1_N1,p_f_2_N1,p_c_1_N1,p_c_2_N1,p_c_3_N1,p_h_1_N1,p_h_2_N1,p_h_3_N1,p_h_4_N1,p_lf_1_N1,p_lb_1_N1,p_f_3_0,p_f_3_1,p_f_3_2,p_lf_1_dec,... 
%p_lf_2_dec,p_lf_3_dec,p_f_4_1_dec,p_f_4_2_dec,p_f_1,p_f_2,f,nl,cfn,dp,K_conv_l1_n1,C_conv_l1_n1,K_conv_l2_n1,C_conv_l2_n1,enc_layers,Em);
        %toc
        
        % Print error (testing) per epoc
        fprintf('\t Epoch : %d  Update: %d Test Loss : %f \n',NE,num_up,test_err);
        
        % save weight ifle
        save_params
    end
    
    % Print error (validation and testing) per epoc
    fprintf(fid,'%d %d %f %f %f \n',NE,num_up,train_err,val_err,test_err);
end
