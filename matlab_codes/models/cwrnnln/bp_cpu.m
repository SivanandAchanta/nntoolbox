function [gWi,gWfr,gbh,gU,gbo,ggn,gbn] = bp_cpu(nl,f,X,Y,ol_mat,sigv,hcm,scm,acm,bacm,pacm,GU,GWfr,Ggn,sl,a_tanh,b_tanh,bby2a,cfn,eyem,tsteps,nng)

% bacward prop
% Compute deltas of output layer weights and biases

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
delnst = (zeros(nl(2),1));
delm = (zeros(sl,nl(2)));
delbm = (zeros(sl,nl(2)));

iemat = GU'*delo';
GWfr = GWfr';
[der_f] = get_derf(nl(end-1),f(end-1),hcm,sl,a_tanh,b_tanh,bby2a);
der_f = der_f';
dhi = 1/nl(2);

zeros_vec = zeros(nl(2),1);

for k = sl:-1:1
    
    lrwsint = find(mod(k,tsteps)==0,1,'last');
    alpha_t = zeros_vec;
    alpha_t(1:lrwsint*nng) = 1;
    
    lrwsint = find(mod(k+1,tsteps)==0,1,'last');
    alpha_tp1 = zeros_vec;
    alpha_tp1(1:lrwsint*nng) = 1;    
    beta_t = 1 - alpha_tp1;

    delst = (GWfr*delnt) + iemat(:,k) + beta_t.*(delnst);
    delt = alpha_t.*der_f(:,k).*(delst);
    delbm(k,:) = delt;

    % bp thru ln
    [bln] = bp_ln(bacm(k,:)',sigv(k),eyem,dhi,Ggn);

    delt = bln'*delt;    
    
    delm(k,:) =  delt;
    delnt = delt;
    delnst = delst;
end

h_0 = zeros(nl(2),1);
hpm = [h_0';hcm(1:end-1,:)];
gWfr = (delm'*hpm);
gbh = sum(delm)';

% Compute gradients of inpu-hidden layer weights
gWi = (delm'*X);

ggn = sum(delbm.*bacm)';
gbn = sum(delbm)';

end

