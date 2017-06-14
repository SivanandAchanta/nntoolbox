function [gWi,gWfr,gbh,Eb] = bp_cpu_rl(nout,f,iemat,hcm,X,p,sl,dir_flag)

Wfr = p.Wfr;
Wi = p.Wi;

% Compute gradients of recurrent weights and biases
delnt = (zeros(nout,1));
delm = (zeros(sl,nout));

h_0 = zeros(nout,1);

if strcmp(dir_flag,'frnn')
    ix_vec = sl:-1:1;
    hpm = [h_0';hcm(1:end-1,:)];
else
    ix_vec = 1:sl;
    hpm = [hcm(2:end,:);h_0'];
end



Wfr = Wfr';
[der_f] = get_derf(nout,f,hcm,sl);
der_f = der_f';

for k = ix_vec
    ie = der_f(:,k).*(Wfr*delnt+iemat(:,k));
    delnt = ie;
    delm(k,:) = delnt;
end


gWfr = (delm'*hpm);
gbh = sum(delm,1)';

% Compute gradients of inpu-hidden layer weights
gWi = (delm'*X);

Eb = delm*Wi;

end