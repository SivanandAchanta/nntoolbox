% Check Validation Loss
if mod(num_up,check_valfreq) == 0
    
    tic
    [train_err] = compute_error(train_batchdata,train_batchtargets,train_clv,train_test_numbats,p_f_1,p_f_2,p_c_1,p_c_2,p_c_3,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,...
        f,nl,cfn,dp,K_conv_l1,C_conv_l1,K_conv_l2,C_conv_l2);
    toc
    
    tic
    [val_err] = compute_error(val_batchdata,val_batchtargets,val_clv,val_numbats,p_f_1,p_f_2,p_c_1,p_c_2,p_c_3,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,...
        f,nl,cfn,dp,K_conv_l1,C_conv_l1,K_conv_l2,C_conv_l2);
    toc
    
    fprintf('Epoch: %d, Train Loss: %f; Val Loss : %f \n',NE,train_err,val_err);
    
    if val_err < best_val_err
        
        best_val_err = val_err;
        best_epoch = NE;
        
        tic
        [test_err] = compute_error(test_batchdata,test_batchtargets,test_clv,test_numbats,p_f_1,p_f_2,p_c_1,p_c_2,p_c_3,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,...
            f,nl,cfn,dp,K_conv_l1,C_conv_l1,K_conv_l2,C_conv_l2);
        toc
        
        % Print error (testing) per epoc
        fprintf('\t Epoch : %d  Update: %d Test Loss : %f \n',NE,num_up,test_err);
        
        % save weight ifle
        save_params
    end
    
    % Print error (validation and testing) per epoc
    fprintf(fid,'%d %d %f %f %f \n',NE,num_up,train_err,val_err,test_err);
end
