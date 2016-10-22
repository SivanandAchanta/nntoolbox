function [W,pmsgW,pmsxW,pdW] = adadelta(rho,eps_hp,mf,gW,pmsgW,pmsxW,pdW,W)

pmsgW  = rho*(pmsgW) + (1-rho)*(gW.^2);
rmsgW  = sqrt(pmsgW + eps_hp); 
rmsxW  = sqrt(pmsxW + eps_hp);

dW     = -(rmsxW./rmsgW).*gW;
pmsxW  = rho*(pmsxW) + (1-rho)*(dW.^2);

pdW    = dW + mf*pdW;
W      = W + pdW;

end