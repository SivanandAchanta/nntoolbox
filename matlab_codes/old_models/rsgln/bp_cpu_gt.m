function [gWi,gWfr,gbh,gU,gWy,gbo,ggn,gbn] = bp_cpu_gt(nl,f,X,Y,ol_mat,sigv,hcm,acm,bacm,pacm,GWy,GU,GWfr,Ggn,sl,a_tanh,b_tanh,bby2a,cfn,eyem)

% bacward prop

[der_f] = get_derf(nl(end),f(end),ol_mat,sl,a_tanh,b_tanh,bby2a);

switch cfn
    case 'ls'
        costder = -(Y - ol_mat)/sl;
        delo = costder.*der_f;
    case  'nll'
        %         costder = -(Out./ol_m);
        delo  = -(Y - ol_mat)/sl;
end

gU = (delo'*hcm);
gbo = sum(delo)';

% Compute gradients of recurrent weights and biases
delnt = (zeros(nl(2),1));
delm = (zeros(sl,nl(2)));
delbm = (zeros(sl,nl(2)));
dhi = (1/nl(2));

iemat = GU'*delo';
GWfr = GWfr';
[der_f] = get_derf(nl(end-1),f(end-1),hcm,sl,a_tanh,b_tanh,bby2a);
der_f = der_f';

for k = sl:-1:1
    ie = der_f(:,k).*(GWfr*delnt+iemat(:,k));
    delbm(k,:) = ie;    
    % bp thru ln
    [bln] = bp_ln(bacm(k,:)',sigv(k),eyem,dhi,Ggn);
    delnt = bln'*ie;
    
    delm(k,:) = delnt;
end

h_0 = zeros(nl(2),1);
hpm = [h_0';hcm(1:end-1,:)];
gWfr = (delm'*hpm);
gbh = sum(delm)';

% Compute gradients of inpu-hidden layer weights
gWi = (delm'*X);

h_0 = zeros(nl(3),1);
ypm = [h_0';Y(1:end-1,:)];
gWy = (delm'*ypm);


ggn = sum(delbm.*bacm)';
gbn = sum(delbm)';

end