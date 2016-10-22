function [W,pmW,pvW] = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gW,pmW,pvW,W)

beta1_t = beta1*(lam^(num_up-1));

pmW     = beta1_t*pmW + (1 - beta1_t)*gW;
pvW     = beta2*pvW + (1 - beta2)*(gW.^2);

mWt_cap = pmW./(1 - beta1^num_up);
vWt_cap = pvW./(1 - beta2^num_up);

W       = W - alpha*mWt_cap./(sqrt(vWt_cap)+eps_hp);

end

