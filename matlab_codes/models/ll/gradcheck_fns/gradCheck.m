if gradCheckFlag
    
    % Compute numerical gradient
    lno = 1;
    argno = 1; [gUn1]  = compute_NumericalGrad(Gpi1,Gpo,X,Y,f,nl,sl,cfn,Gpi1.U,argno,gpu_flag,lno,dm1);
    argno = argno + 1; [gbun1]  = compute_NumericalGrad(Gpi1,Gpo,X,Y,f,nl,sl,cfn,Gpi1.bu,argno,gpu_flag,lno,dm1);

    lno = lno + 1;
    argno = 1; [gUn2]  = compute_NumericalGrad(Gpi1,Gpo,X,Y,f,nl,sl,cfn,Gpo.U,argno,gpu_flag,lno,dm1);
    argno = argno + 1; [gbun2]  = compute_NumericalGrad(Gpi1,Gpo,X,Y,f,nl,sl,cfn,Gpo.bu,argno,gpu_flag,lno,dm1);
    
    % Compute the difference between numerical and back-prop gradients
         
    [dgUn1] = compute_gradDiff_fn(gUn1,Gpi1.gU,'gU1');
    [dgbun1] = compute_gradDiff_fn(gbun1,Gpi1.gbu,'gbu1');

    [dgUn2] = compute_gradDiff_fn(gUn2,Gpo.gU,'gU2');
    [dgbun2] = compute_gradDiff_fn(gbun2,Gpo.gbu,'gbu2');
    
    % pause
    
end
