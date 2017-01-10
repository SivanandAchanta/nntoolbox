function [gWi,gWfr,gbh,gU,gUfr,gbo] = bp_cpu_gt(nl,f,X,Y,ol_mat,hcm,GUfr,GU,GWfr,sl,a_tanh,b_tanh,bby2a,cfn)

% bacward prop
% Compute deltas of output layer weights and biases

[der_f] = get_derf(nl(end),f(end),ol_mat,sl,a_tanh,b_tanh,bby2a);

switch cfn
    case 'ls'
        costder = -(Y - ol_mat)/sl;
        delo = costder;
    case  'nll'
        %         costder = -(Out./ol_m);
        delo  = -(Y - ol_mat)/sl;
end

% Compute gradients of recurrent weights and biases
delm = delo;
h_0 = zeros(nl(3),1);
ypm = [h_0';Y(1:end-1,:)];
gUfr = (delm'*ypm);
gbo = sum(delm)';

% Compute gradients of inpu-hidden layer weights
gU = (delm'*hcm);


% Compute gradients of recurrent weights and biases
iemat = GU'*delm';
delnt = (zeros(nl(2),1));
delm = (zeros(sl,nl(2)));

GWfr = GWfr';
[der_f] = get_derf(nl(end-1),f(end-1),hcm,sl,a_tanh,b_tanh,bby2a);
der_f = der_f';

for k = sl:-1:1
    ie = der_f(:,k).*(GWfr*delnt+iemat(:,k));
    delnt = ie;
    delm(k,:) = delnt;
end

h_0 = zeros(nl(2),1);
hpm = [h_0';hcm(1:end-1,:)];
gWfr = (delm'*hpm);
gbh = sum(delm)';

% Compute gradients of inpu-hidden layer weights
gWi = (delm'*X);

end