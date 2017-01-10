function [gWn] = compute_NumericalGrad(W,b,X,Y,fl,nl,nh,nlv,wtl,btl,a_tanh,b_tanh,bs,cfn,W_toCheck,argno,gpu_flag)

W1 = W;
W2 = b;

% perturbation magnitude at x
h = 1e-6;

% Compute the numerical gradient
[nr,nc]  = size(W_toCheck);

if gpu_flag
    gWn      = gpuArray(zeros(nr,nc));
else
    gWn      = (zeros(nr,nc));
end

for cc = 1:nc
    for cr = 1:nr
        
        switch argno
            case 1
                W1 = W_toCheck; W1(cr,cc) = W1(cr,cc) + h;
            case 2
                W2 = W_toCheck; W2(cr,cc) = W2(cr,cc) + h;                
        end
        
        
        [f_xph] = compute_Fofx(X,Y,W1,W2,fl,nl,nh,nlv,wtl,btl,a_tanh,b_tanh,bs,cfn);
        
        switch argno
            case 1
                W1 = W_toCheck; W1(cr,cc) = W1(cr,cc) - h;
            case 2
                W2 = W_toCheck; W2(cr,cc) = W2(cr,cc) - h;                
        end
        
        [f_xnh] = compute_Fofx(X,Y,W1,W2,fl,nl,nh,nlv,wtl,btl,a_tanh,b_tanh,bs,cfn);
        
        gWn(cr,cc) = (f_xph-f_xnh)/(2*h);
    end
end


end