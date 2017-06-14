if gradCheckFlag
    
    % Compute numerical gradient
    argno = 1; [gWzn] = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,Wz,argno,dnn_flag);
    argno = argno + 1; [gRzn] = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,Rz,argno,dnn_flag);
    argno = argno + 1; [gbzn] = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,bz,argno,dnn_flag);
    
    argno = argno + 1; [gWin] = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,Wi,argno,dnn_flag);    
    argno = argno + 1; [gRin]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,Ri,argno,dnn_flag);
    argno = argno + 1; [gpin]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,pi,argno,dnn_flag);
    argno = argno + 1; [gbin]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,bi,argno,dnn_flag);
    
    argno = argno + 1; [gWfn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,Wf,argno,dnn_flag);    
    argno = argno + 1; [gRfn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,Rf,argno,dnn_flag);
    argno = argno + 1; [gpfn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,pf,argno,dnn_flag);
    argno = argno + 1; [gbfn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,bf,argno,dnn_flag);
    
    argno = argno + 1; [gWon]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,Wo,argno,dnn_flag);    
    argno = argno + 1; [gRon]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,Ro,argno,dnn_flag);
    argno = argno + 1; [gpon]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,po,argno,dnn_flag);
    argno = argno + 1; [gbon]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,bo,argno,dnn_flag);
    
    argno = argno + 1; [gUn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,U,argno,dnn_flag);
    argno = argno + 1; [gbun]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,bu,argno,dnn_flag);
    
    argno = argno + 1; [gWn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,W,argno,dnn_flag);
    argno = argno + 1; [gbn]  = compute_NumericalGrad(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,W,b,X,Y,f,nl,nl_dnn,f_dnn,nh_dnn,wtl_dnn,btl_dnn,nlv_dnn,sl,cfn,b,argno,dnn_flag);
    
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
    
    [dgWn] = compute_gradDiff_fn(gWn,gW,'gW');
    [dgbn] = compute_gradDiff_fn(gbn,gb,'gb');
    
    pause
    
end