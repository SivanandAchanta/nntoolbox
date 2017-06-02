% fp and bp for each batch
for i = 1:train_numbats
    
    num_up = num_up + 1;
    
    % get data
    [X,Y,sl] = get_XY_seqver(train_batchdata, train_batchtargets, train_clv, rp, i, gpu_flag);
    
    % fp
    [hcm,ol_mat] = fp_cpu(X,GWi,GWfr,GU,Gbh,Gbo,f,nl,a_tanh,b_tanh,sl);
    
    % bp
    [gWi,gWfr,gbh,gU,gbo] = bp_cpu(nl,f,X,Y,ol_mat,hcm,GU,GWfr,sl,a_tanh,b_tanh,bby2a,cfn);
    
    % clip the gradients above a threshold
    if gc_flag
        clip_grad
    end
    
    if gradCheckFlag
        gradCheck
    end
    
    % Update Params using Appropriate SGD Method
    update_params
    
    compute_train_val_test_error
    
    if isnan(val_err)
        break;
    end
    
end