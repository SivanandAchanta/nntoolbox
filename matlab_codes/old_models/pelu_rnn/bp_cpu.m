function [gWi,gWfr,gbh,gU,gbo,gap,gbp] = bp_cpu(nl,f,X,Y,ol_mat,hcm,GU,GWfr,Gapelu,Gbpelu,sl,a_tanh,b_tanh,bby2a,cfn)

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
gbo = sum(delo,1)';

% Compute gradients of recurrent weights and biases
delnt = (zeros(nl(2),1));
delnt1 = (zeros(nl(2),1));
delm = (zeros(sl,nl(2)));
delbm = (zeros(sl,nl(2)));

iemat = GU'*delo';
GWfr = GWfr';
[der_f,dhcm_ap,dhcm_bp] = get_derf(nl(end-1),f(end-1),hcm,sl,a_tanh,b_tanh,bby2a,Gapelu,Gbpelu);
der_f = der_f';

for k = sl:-1:1
    
    ie1 = (GWfr*delnt+iemat(:,k));
    delnt1 = ie1;
    delbm(k,:) = delnt1;

    ie = der_f(:,k).*(GWfr*delnt+iemat(:,k));
    delnt = ie;
    delm(k,:) = delnt;
    
end

h_0 = zeros(nl(2),1);
hpm = [h_0';hcm(1:end-1,:)];
gWfr = (delm'*hpm);
gbh = sum(delm,1)';

% Compute gradients of inpu-hidden layer weights
gWi = (delm'*X);

gap = sum(delbm.*dhcm_ap,1)';
gbp = sum(delbm.*dhcm_bp,1)';

end