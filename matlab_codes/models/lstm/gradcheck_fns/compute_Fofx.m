function [J] = compute_Fofx(X,Y,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,U,bu,nl,sl,f,cfn)


% Forward Pass
[~,~,~,~,~,~,~,ym] = fp_lstm(X,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,nl,sl,f,U,bu);

% Cost Funtion
switch cfn
    case 'ls'
        J   = 0.5*mean(sum(power((Y - ym),2),2));
    case  'nll'
        J   = mean(sum((-Y.*log(ym)),2));
        
end


end