% Forward Pass
[zm,im,fm,cm,om,hcm,hm,ym] = fp_lstm(X,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,nl,sl,f,U,bu);

% Error at Output Layer
switch cfn
    case 'ls'
        E   = -(Y - ym)/sl;
    case  'nll'
        E   = -(Y - ym)/sl;
end

% Backprop
gU = (E'*hm);
gbu = sum(E)';
Eb = U'*E';

[dhm,dom,dcm,dfm,dim,dzm] = bp_lstm(Eb,Rz,Ri,pi,Rf,pf,Ro,po,zm,im,fm,cm,om,hcm,nl,sl);
[gWz,gRz,gbz,gWi,gRi,gpi,gbi,gWf,gRf,gpf,gbf,gWo,gRo,gpo,gbo] = gradients_lstm(X,hm,cm,dom,dfm,dim,dzm,nl);
