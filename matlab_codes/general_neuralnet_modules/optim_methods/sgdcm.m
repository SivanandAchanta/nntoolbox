function [W,pdW] = sgdcm(lr,mf,gW,pdW,W)

dW  = -lr*gW;
pdW = dW + mf*pdW;
W   = W + pdW;

end