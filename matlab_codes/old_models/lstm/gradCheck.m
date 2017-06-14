if gradCheckFlag
    
    % Compute numerical gradient
    argno = 1; [gWzn] = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Wz,argno,gpu_flag);
    argno = argno + 1; [gRzn] = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Rz,argno,gpu_flag);
    argno = argno + 1; [gbzn] = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,bz,argno,gpu_flag);
    
    argno = argno + 1; [gWin] = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Wi,argno,gpu_flag);    
    argno = argno + 1; [gRin]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Ri,argno,gpu_flag);
    argno = argno + 1; [gpin]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,pi,argno,gpu_flag);
    argno = argno + 1; [gbin]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,bi,argno,gpu_flag);
    
    argno = argno + 1; [gWfn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Wf,argno,gpu_flag);    
    argno = argno + 1; [gRfn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Rf,argno,gpu_flag);
    argno = argno + 1; [gpfn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,pf,argno,gpu_flag);
    argno = argno + 1; [gbfn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,bf,argno,gpu_flag);
    
    argno = argno + 1; [gWon]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Wo,argno,gpu_flag);    
    argno = argno + 1; [gRon]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Ro,argno,gpu_flag);
    argno = argno + 1; [gpon]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,po,argno,gpu_flag);
    argno = argno + 1; [gbon]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,bo,argno,gpu_flag);
    
    argno = argno + 1; [gUn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,U,argno,gpu_flag);
    argno = argno + 1; [gbun]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,bu,argno,gpu_flag);
    
    % Compute the difference between numerical and back-prop gradients
    [dgWzn] = compute_gradDiff_fn(gWzn,gWz,'gWz');
    [dgWin] = compute_gradDiff_fn(gWin,gWi,'gWi');
    [dgWfn] = compute_gradDiff_fn(gWfn,gWf,'gWf');
    [dgWon] = compute_gradDiff_fn(gWon,gWo,'gWo');
    
    [dgRzn] = compute_gradDiff_fn(gRzn,gRz,'gRz');
    [dgRin] = compute_gradDiff_fn(gRin,gRi,'gRi');
    [dgRfn] = compute_gradDiff_fn(gRfn,gRf,'gRf');
    [dgRon] = compute_gradDiff_fn(gRon,gRo,'gRo');
    
    [dgbzn] = compute_gradDiff_fn(gbzn,gbz,'gbz');
    [dgbin] = compute_gradDiff_fn(gbin,gbi,'gbi');
    [dgbfn] = compute_gradDiff_fn(gbfn,gbf,'gbf');
    [dgbon] = compute_gradDiff_fn(gbon,gbo,'gbo');
    
    [dgpin] = compute_gradDiff_fn(gpin,gpi,'gpi');
    [dgpfn] = compute_gradDiff_fn(gpfn,gpf,'gpf');
    [dgpon] = compute_gradDiff_fn(gpon,gpo,'gpo');
    
    [dgUn] = compute_gradDiff_fn(gUn,gU,'gU');
    [dgbun] = compute_gradDiff_fn(gbun,gbu,'gbu');
    
    pause
    
end