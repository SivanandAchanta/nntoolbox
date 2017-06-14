function [gWi,gWfr,gbh,gU,gWy,gbo] = bp_cpu_pt(nl,f,X,Y,ol_mat,hcm,GWy,GU,GWfr,sl,a_tanh,b_tanh,bby2a,cfn)

% bacward prop

[der_fy] = get_derf(nl(end),f(end),ol_mat,sl,a_tanh,b_tanh,bby2a);
der_fy = der_fy';

switch cfn
    case 'ls'
        costder = -(Y - ol_mat)/sl;
        delo = costder;
    case  'nll'
        %         costder = -(Out./ol_m);
        delo  = -(Y - ol_mat)/sl;
end

% Compute gradients of recurrent weights and biases
delnt = (zeros(nl(2),1));
delm = (zeros(sl,nl(2)));
dely = (zeros(sl,nl(3)));

iematy = delo';

GWy = GWy';
GWfr = GWfr';
[der_fh] = get_derf(nl(end-1),f(end-1),hcm,sl,a_tanh,b_tanh,bby2a);
der_fh = der_fh';

for k = sl:-1:1

    ie = der_fy(:,k).*(GWy*delnt+iematy(:,k));
    dely(k,:) = ie';
    
    ieh = GU'*ie;

    ie = der_fh(:,k).*(GWfr*delnt+ieh);
    delnt = ie;
    delm(k,:) = delnt;
end


h_0 = zeros(nl(3),1);
ypm = [h_0';ol_mat(1:end-1,:)];
gU = (dely'*hcm);
gWy = (delm'*ypm);
gbo = sum(dely)';

h_0 = zeros(nl(2),1);
hpm = [h_0';hcm(1:end-1,:)];
gWfr = (delm'*hpm);
gbh = sum(delm)';

% Compute gradients of inpu-hidden layer weights
gWi = (delm'*X);

end