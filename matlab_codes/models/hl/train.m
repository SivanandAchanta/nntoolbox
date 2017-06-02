% fp and bp for each batch
for i = 1:train_numbats
    
    num_up = num_up + 1;
    
    % get data
    [X,Y,sl] = get_XY_seqver(train_batchdata, train_batchtargets, train_clv, rp, i, gpu_flag);
    
    % fp
    [tm1,htm1,hm1] = fp_hl(X,Gpi1);
    [tm2,htm2,hm2] = fp_hl(hm1,Gpi2);
    ol_mat = fp_cpu_ll(hm2,Gpo,f(end));
    
    % bp
    switch cfn
        case 'ls'
            E = -(Y - ol_mat)/sl;
        case  'nll'
            E  = -(Y - ol_mat)/sl;
    end
    
    [Gpo.gU,Gpo.gbu,Eb] = bp_cpu_ll(nl(end),f(end),E,ol_mat,hm2,Gpo.U,sl);
    [Gpi2.gW,Gpi2.gb,Gpi2.gWt,Gpi2.gbt,Eb] = bp_hl(nl(3),sl,Eb,Gpi2,htm2,tm2,hm1);
    [Gpi1.gW,Gpi1.gb,Gpi1.gWt,Gpi1.gbt,Eb] = bp_hl(nl(2),sl,Eb,Gpi1,htm1,tm1,X);
    
    if gradCheckFlag
        gradCheck
    end
    
    % Update Params using Appropriate SGD Method
    [Gpi1] = update_params_hl(Gpi1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2);
    [Gpi2] = update_params_hl(Gpi2,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2);
    [Gpo] = update_params_ll(Gpo,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2);
    
    compute_train_val_test_error
    
    if isnan(val_err)
        break;
    end
    
end