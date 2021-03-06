if gradCheckFlag
    
    % Compute numerical gradient
    argno = 1; [gWin] = compute_NumericalGrad(GWi,GWfr,Gbh,GU,Gbo,Ggn,Gbn,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,GWi,argno,gpu_flag,lr_fw,dr_fw);
    argno = argno + 1; [gWfrn]  = compute_NumericalGrad(GWi,GWfr,Gbh,GU,Gbo,Ggn,Gbn,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,GWfr,argno,gpu_flag,lr_fw,dr_fw);
    argno = argno + 1; [gbhn]  = compute_NumericalGrad(GWi,GWfr,Gbh,GU,Gbo,Ggn,Gbn,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Gbh,argno,gpu_flag,lr_fw,dr_fw);
    argno = argno + 1; [gUn]  = compute_NumericalGrad(GWi,GWfr,Gbh,GU,Gbo,Ggn,Gbn,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,GU,argno,gpu_flag,lr_fw,dr_fw);
    argno = argno + 1; [gbon]  = compute_NumericalGrad(GWi,GWfr,Gbh,GU,Gbo,Ggn,Gbn,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Gbo,argno,gpu_flag,lr_fw,dr_fw);
    argno = argno + 1; [ggnn]  = compute_NumericalGrad(GWi,GWfr,Gbh,GU,Gbo,Ggn,Gbn,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Ggn,argno,gpu_flag,lr_fw,dr_fw);
    argno = argno + 1; [gbnn]  = compute_NumericalGrad(GWi,GWfr,Gbh,GU,Gbo,Ggn,Gbn,X,Y,f,nl,a_tanh,b_tanh,sl,cfn,Gbn,argno,gpu_flag,lr_fw,dr_fw);
    
    % Compute the difference between numerical and back-prop gradients
    [dgWin] = compute_gradDiff_fn(gWin,gWi,'gWi');
    [dgWfrn] = compute_gradDiff_fn(gWfrn,gWfr,'gWfr');
    [dgbhn] = compute_gradDiff_fn(gbhn,gbh,'gbh');
    [dgUn] = compute_gradDiff_fn(gUn,gU,'gU');
    [dgbon] = compute_gradDiff_fn(gbon,gbo,'gbo');
    [dggnn] = compute_gradDiff_fn(ggnn,ggn,'ggn');
    [dgbnn] = compute_gradDiff_fn(gbnn,gbn,'gbn');
    
    pause
    
end