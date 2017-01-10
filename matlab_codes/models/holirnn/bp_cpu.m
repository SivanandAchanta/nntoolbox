function [gWi,gWfr,gbh,gU,gbo] = bp_cpu(nl,f,X,Y,ol_mat,hcm,scm,GU,GWfr,sl,a_tanh,b_tanh,bby2a,cfn)

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
delnst = (zeros(nl(2),1));
delm = (zeros(sl,nl(2)));

iemat = GU'*delo';
GWfr = GWfr';
[der_f] = get_derf(nl(end-1),f(end-1),scm,sl,a_tanh,b_tanh,bby2a);
der_f = der_f';

% lam = 0.99;
% alpha = 0.7;
% beta = 1 - alpha;
% alpha_vec = alpha*ones(nl(2),1);
% beta_vec = beta*ones(nl(2),1);
% alpha_vec = linspace(0,alpha,nl(2))';
% beta_vec = 1 - alpha_vec;


for k = sl:-1:1
    
%     delst = (GWfr*delnt) + iemat(:,k)  + beta_vec.*(delnst);
%     delt = alpha_vec.*der_f(:,k).*(delst);

    delst = (GWfr*delnt) + iemat(:,k)  + (delnst);
    delt = der_f(:,k).*(delst);

    delm(k,:) =  delt;
    delnt = delt;
    delnst = delst;    
end

% delm(2,:) = (der_f(:,2).*iemat(:,2))';
% 
% delm(1,:) = (der_f(:,1).*(GWfr*delm(2,:)' + iemat(:,1) + iemat(:,2)))';

h_0 = zeros(nl(2),1);
hpm = [h_0';hcm(1:end-1,:)];
gWfr = (delm'*hpm);
gbh = sum(delm,1)';

% Compute gradients of inpu-hidden layer weights
gWi = (delm'*X);

end