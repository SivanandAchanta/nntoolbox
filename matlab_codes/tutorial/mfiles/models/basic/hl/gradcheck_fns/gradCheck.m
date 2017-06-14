if gradCheckFlag
    
    % Compute numerical gradient
    lno = 1;
    argno = 1; [gWn1] = compute_NumericalGrad(Gpi1,Gpi2,Gpo,X,Y,f,nl,sl,cfn,Gpi1.W,argno,gpu_flag,lno);
    argno = argno + 1; [gWtn1]  = compute_NumericalGrad(Gpi1,Gpi2,Gpo,X,Y,f,nl,sl,cfn,Gpi1.Wt,argno,gpu_flag,lno);
    argno = argno + 1; [gbn1]  = compute_NumericalGrad(Gpi1,Gpi2,Gpo,X,Y,f,nl,sl,cfn,Gpi1.b,argno,gpu_flag,lno);
    argno = argno + 1; [gbtn1]  = compute_NumericalGrad(Gpi1,Gpi2,Gpo,X,Y,f,nl,sl,cfn,Gpi1.bt,argno,gpu_flag,lno);
    
    lno = lno + 1;
    argno = 1; [gWn2] = compute_NumericalGrad(Gpi1,Gpi2,Gpo,X,Y,f,nl,sl,cfn,Gpi2.W,argno,gpu_flag,lno);
    argno = argno + 1; [gWtn2]  = compute_NumericalGrad(Gpi1,Gpi2,Gpo,X,Y,f,nl,sl,cfn,Gpi2.Wt,argno,gpu_flag,lno);
    argno = argno + 1; [gbn2]  = compute_NumericalGrad(Gpi1,Gpi2,Gpo,X,Y,f,nl,sl,cfn,Gpi2.b,argno,gpu_flag,lno);
    argno = argno + 1; [gbtn2]  = compute_NumericalGrad(Gpi1,Gpi2,Gpo,X,Y,f,nl,sl,cfn,Gpi2.bt,argno,gpu_flag,lno);
        
    lno = lno + 1;
    argno = 1; [gUn]  = compute_NumericalGrad(Gpi1,Gpi2,Gpo,X,Y,f,nl,sl,cfn,Gpo.U,argno,gpu_flag,lno);
    argno = argno + 1; [gbun]  = compute_NumericalGrad(Gpi1,Gpi2,Gpo,X,Y,f,nl,sl,cfn,Gpo.bu,argno,gpu_flag,lno);
    
    % Compute the difference between numerical and back-prop gradients
    [dgWn1] = compute_gradDiff_fn(gWn1,Gpi1.gW,'gW1');
    [dgWtn1] = compute_gradDiff_fn(gWtn1,Gpi1.gWt,'gWt1');
    [dgbn1] = compute_gradDiff_fn(gbn1,Gpi1.gb,'gb1');
    [dgbtn1] = compute_gradDiff_fn(gbtn1,Gpi1.gbt,'gbt1');
    
    [dgWn2] = compute_gradDiff_fn(gWn2,Gpi2.gW,'gW2');
    [dgWtn2] = compute_gradDiff_fn(gWtn2,Gpi2.gWt,'gWt2');
    [dgbn2] = compute_gradDiff_fn(gbn2,Gpi2.gb,'gb2');
    [dgbtn2] = compute_gradDiff_fn(gbtn2,Gpi2.gbt,'gbt2');
        
    [dgUn] = compute_gradDiff_fn(gUn,Gpo.gU,'gU');
    [dgbun] = compute_gradDiff_fn(gbun,Gpo.gbu,'gbu');
    
    pause
    
end