function [gWi,gWfr,gbh,gU,gbo] = bp_cpu(nl,f,X,Y,ol_mat,hcm,alpham,scm,GU,GWfr,sl,a_tanh,b_tanh,bby2a,cfn,lr_fw,dr_fw)

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
delnt_fw = (zeros(nl(2),1));
delm = (zeros(sl,nl(2)));

iemat = GU'*delo';
GWfr = GWfr';

[der_f] = get_derf(nl(end-1),f(end-1),hcm,sl,a_tanh,b_tanh,bby2a);
der_f = der_f';

[der_fi] = get_derf(nl(end-1),f(end-1),scm,sl,a_tanh,b_tanh,bby2a);
der_fi = der_fi';

h_0 = zeros(nl(2),1);
hpm = [h_0';hcm(1:end-1,:)];

% for k = sl
k = sl;
delm(k,:) = delm(k,:) + (GWfr*(delnt + delnt_fw))' + (iemat(:,k))';
delm(k,:) = der_f(:,k)'.*(delm(k,:));
delnt = delm(k,:)';

dr_vec = (dr_fw).^(k-(1:k));
dhcm = bsxfun(@times,hcm(1:k-1,:),dr_vec(1:k-1)');
hth = (hcm(1:k-1,:)'*dhcm(1:k-1,:));
delnt_fw = lr_fw*der_fi(:,k).*((hth)'*delnt);
delm(k,:) = delm(k,:) + delnt_fw';


for k = sl-1:-1:2
      
    % case 3
    % Compute Da
    Da = lr_fw*alpham(k+1,:)'*delnt';
    % Compute Dg
    st = scm(k+1,:);
    d_beta = hcm*(delnt*st);    
    d_beta(k+1:end,:) = 0;
    d_beta(1:k,:) = bsxfun(@times,d_beta(1:k,:),dr_vec(1:k)');    
    Dg = lr_fw*(d_beta);
    % Sum Da and Dg to past error signals
    delm = delm + Da + Dg;
    
    % compute delt
    delm(k,:) = delm(k,:) + (GWfr*(delnt + delnt_fw))' + (iemat(:,k))';
    delm(k,:) = der_f(:,k)'.*(delm(k,:));
    delnt = delm(k,:)';
    
    % add delfwt to delt
    dr_vec = (dr_fw).^(k-(1:k));
    dhcm = bsxfun(@times,hcm(1:k-1,:),dr_vec(1:k-1)');
    hth = (hcm(1:k-1,:)'*dhcm(1:k-1,:));
    delnt_fw = lr_fw*der_fi(:,k).*((hth)'*delnt);
    delm(k,:) = delm(k,:) + delnt_fw';
    
end

% for k = 1
k = 1;
% compute Da
Da = lr_fw*alpham(k+1,:)'*delnt';
% compute Dg
st = scm(k+1,:);
d_beta = hcm*(delnt*st);
d_beta(k+1:end,:) = 0;
d_beta(1:k,:) = bsxfun(@times,d_beta(1:k,:),dr_vec(1:k)');
Dg = lr_fw*(d_beta);
% add to the past time steps
delm = delm + Da + Dg;
% delt computation
delm(k,:) = delm(k,:) + (GWfr*(delnt + delnt_fw))' + (iemat(:,k))';
delm(k,:) = der_f(:,k)'.*(delm(k,:));



% Compute gradients of hidden layer params
gWfr = (delm'*hpm);
gbh = sum(delm,1)';

% Compute gradients of input-hidden layer weights
gWi = (delm'*X);


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Case 1 - Method 1

% t1 = delm(sl:-1:k,:)';
% t2 = (alpham(sl:-1:k,k));
% %(delm(sl:-1:k,:)')*(alpham(sl:-1:k,k))
% delt = der_f(:,k).*(GWfr*delnt + iemat(:,k) + t1*t2);
%
% delm(k,:) = delt';
% delnt = delt;
% delnt'
% delm(k,:)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Case 1 - Method 2

% if k < sl
%     delm = delm + alpham(k+1,:)'*delnt';
% end

% delm(k,:) = delm(k,:) + (GWfr*delnt)' + (iemat(:,k))';
% delm(k,:) = der_f(:,k)'.*(delm(k,:));
% delnt = delm(k,:)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Case2 - Method 2

% hc_s = ones(1,nl(2));
% hc_s = (0.9.^(1:length(hc_s)));
% d_beta = hcm*(delnt*hc_s);
% if k < sl
%     d_beta(k+1:end,:) = 0;
%     delm = delm + alpham(k+1,:)'*delnt' + (d_beta);
% end
%
% delm(k,:) = delm(k,:) + (GWfr*delnt)' + (iemat(:,k))';
% delm(k,:) = der_f(:,k)'.*(delm(k,:));
% delnt = delm(k,:)';

