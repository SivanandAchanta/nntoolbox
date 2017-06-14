% fp and bp for each batch
for i = 1:train_numbats
    
    num_up = num_up + 1;
    
    % get data
    [X,Y,sl] = get_XY_seqver(train_batchdata, train_batchtargets, train_clv, rp, i, gpu_flag);
    
    % fp
    hcm = fp_cpu_rl(X,Gpi1,f,nl,sl);
    ol_mat = fp_cpu_ll(hcm,Gpo,f(end));
    
    % bp
    [Gpo.gU,Gpo.gbu,Eb] = bp_cpu_ll(nl(end),f(end),Y,ol_mat,hcm,Gpo.U,sl,cfn);
    [Gpi1.gWi,Gpi1.gWfr,Gpi1.gbh,Eb] = bp_cpu_rl(nl(2),f(1),Eb',hcm,X,Gpi1,sl);
    
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