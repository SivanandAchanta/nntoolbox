% fp and bp for each batch
for i = 1:train_numbats
    
    num_up = num_up + 1;
    
    % get data
    [X,Y,sl] = get_XY_seqver(train_batchdata, train_batchtargets, train_clv, rp, i, gpu_flag);
    
    % fp
    fp_model
    bp_model
    
    % clip the gradients above a threshold
    if gc_flag
        clip_grad
    end
    
    %     if gradCheckFlag
    %         gradCheck
    %     end
    
    % Update Params using Appropriate SGD Method
    update_params
    
    compute_train_val_test_error
    
    if isnan(val_err)
        break;
    end
    
end