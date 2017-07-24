function [bm,sm,cm] = fp_att(H,X,Y,p1,p0,nout,sl_dec)

bm = (zeros(sl_dec,size(X,2)));
sm = (zeros(sl_dec,size(X,2)));
cm = (zeros(sl_dec,nout));

Wa = p1.U;
va = p0.bu;

for k = 1:sl_dec
    % alignment model
    ac = bsxfun(@plus,X,Wa*Y(:,k));
    gm = get_actf('N',ac);
    e_kj = va'*gm;
    
    % soft max
    ee_kj = exp(e_kj);
    alpha_k = ee_kj/sum(ee_kj);
    
    % linear combination of annotation matrix
    c_k = H*alpha_k';

    % store vectors for T time steps
    bm(k,:) = e_kj;
    sm(k,:) = alpha_k;

    
    cm(k,:) = c_k';
end
