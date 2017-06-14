function [gU,Eb] = bp_cpu_ll_ow(nout,f,delo,ol_mat,hcm,U,sl)

% bacward prop
% Compute deltas of output layer weights and biases

[der_f] = get_derf(nout,f,ol_mat,sl);
if ~strcmp(f,'M')
    delo = der_f.*delo;
end

gU = (delo'*hcm);

Eb = delo*U;
end
