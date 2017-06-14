if gradCheckFlag
    
    % Compute numerical gradient
    argno = 1; [gWn] = compute_NumericalGrad(GW,Gb,X,Y,fl,nl,nh,nlv,wtl,btl,a_tanh,b_tanh,bs,cfn,GW,argno,gpu_flag);
    argno = argno + 1; [gbn]  = compute_NumericalGrad(GW,Gb,X,Y,fl,nl,nh,nlv,wtl,btl,a_tanh,b_tanh,bs,cfn,Gb,argno,gpu_flag);
    
    % Compute the difference between numerical and back-prop gradients
    [dgWn] = compute_gradDiff_fn(gWn,gW,'gW');
    [dgbn] = compute_gradDiff_fn(gbn,gb,'gb');
    
    dgbn
    dgWn
    pause
    
end