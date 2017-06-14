function [p0,p1,p2,Eb1,Eb2] = bp_att(Eb,sm,H,X,Y,p1,p2,p0,nin1,nin2,nout,sl_dec,sl_enc)

% H - sl_enc x 2*dh
% Eb - sl_dec x 2*dh

dsm = H*Eb'; % sl_enc x sl_dec

Wa = p2.U;
va = p0.bu;

gva = zeros(nout,1);
gU1 = zeros(nout,2*nin2);
gbu1 = zeros(nout,1);
gU2 = zeros(nout,nin1);
Eb1 = zeros(sl_dec,nin1);
Eb2 = zeros(sl_enc,2*nin2);

% der
for i = sl_dec:-1:1
    dsv = dsm(:,i);
    der_f_sm = (diag(sm(i,:)) - sm(i,:)'*sm(i,:)); % sle x sle
    db = der_f_sm*dsv; % sle x 1
    
    % alignment model
    ac = bsxfun(@plus,X,Wa*Y(:,i));
    gm = get_actf('N',ac); % da x sle
    
    gva = gva + (gm*db); % da x 1
    
    
    dg = (db*va')'; %  da x sle
    der_f_cg = get_derf(nout,'N',gm,sl_enc); % da x sle
    dg = dg.*der_f_cg; % da x sle
    
    gU1 = gU1 + dg*H; % da x 2*dh
    gbu1 = gbu1 + sum(dg,2); % da x 1
    
    S = repmat(Y(:,i)',sl_enc,1); % sle x dh_dec
    gU2 = gU2 + dg*S; % da x dh_dec
    
    Eb1(i,:) = sum(dg'*p2.U,1)';
    Eb2 = Eb2 + dg'*p1.U + sm(i,:)'*Eb(i,:);
    
end

p0.gbu = gva;
p1.gU = gU1;
p1.gbu = gbu1;
p2.gU = gU2;



end