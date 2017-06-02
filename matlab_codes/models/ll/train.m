% fp and bp for each batch
for i = 1:train_numbats
    
    num_up = num_up + 1;
    
    % get data
    [X,Y,sl] = get_XY_seqver(train_batchdata, train_batchtargets, train_clv, rp, i, gpu_flag);
    
    % fp
    hm1 = fp_cpu_ll(X,Gpi1,f(1));
    [hm1,dm1] = fp_dropout(hm1,dp(1),nl(2));
    ol_mat = fp_cpu_ll(hm1,Gpo,f(end));
    
    % bp
    switch cfn
        case 'ls'
            E = -(Y - ol_mat)/sl;
        case  'nll'
            E  = -(Y - ol_mat)/sl;
    end
    
    
    [Gpo.gU,Gpo.gbu,Eb] = bp_cpu_ll(nl(end),f(end),E,ol_mat,hm1,Gpo.U,sl);     
    Eb = bp_dropout(Eb,dm1);
    [Gpi1.gU,Gpi1.gbu,Eb] = bp_cpu_ll(nl(2),f(1),Eb,hm1,X,Gpi1.U,sl);
    
    if gradCheckFlag
        gradCheck
    end
    
    % Update Params using Appropriate SGD Method
    [Gpi1] = update_params_ll(Gpi1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2);
    [Gpo] = update_params_ll(Gpo,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2);
    
    compute_train_val_test_error
    
    if isnan(val_err)
        break;
    end
    
end
