function [gWn] = compute_NumericalGrad(p1,p2,X,Y,f,nl,sl,cfn,W_toCheck,argno,gpu_flag,lno,dm1)

switch lno
        
    case 1
        W1 = p1.U;
        W2 = p1.bu;
    case 2
        W1 = p2.U;
        W2 = p2.bu;
        
end



% perturbation magnitude at x
h = 1e-5;

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
        
        switch lno
                
            case 1
                p1.U  = W1;
                p1.bu = W2;
                
            case 2
                p2.U  = W1;
                p2.bu = W2;
                
        end
        
        
        [f_xph] = compute_Fofx(X,Y,p1,p2,f,nl,sl,cfn,dm1);
        
        switch argno
            case 1
                W1 = W_toCheck; W1(cr,cc) = W1(cr,cc) - h;
            case 2
                W2 = W_toCheck; W2(cr,cc) = W2(cr,cc) - h;
                
        end
        
        switch lno
                
            case 1
                p1.U  = W1;
                p1.bu = W2;
                
            case 2
                p2.U  = W1;
                p2.bu = W2;
                
        end
        
        [f_xnh] = compute_Fofx(X,Y,p1,p2,f,nl,sl,cfn,dm1);
        
        gWn(cr,cc) = (f_xph-f_xnh)/(2*h);
    end
end


end
