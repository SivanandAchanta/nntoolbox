function [gWn] = compute_NumericalGrad(p1,p2,p3,X,Y,f,nl,sl,cfn,W_toCheck,argno,gpu_flag,lno)

switch lno
    case 1
        W1 = p1.W;
        W2 = p1.Wt;
        W3 = p1.b;
        W4 = p1.bt;
        
    case 2
        W1 = p2.W;
        W2 = p2.Wt;
        W3 = p2.b;
        W4 = p2.bt;
        
    case 3
        W1 = p3.U;
        W2 = p3.bu;
        
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
            case 3
                W3 = W_toCheck; W3(cr,cc) = W3(cr,cc) + h;
            case 4
                W4 = W_toCheck; W4(cr,cc) = W4(cr,cc) + h;
                
        end
        
        switch lno
            case 1
                p1.W  = W1;
                p1.Wt = W2;
                p1.b  = W3;
                p1.bt = W4;
                
            case 2
                p2.W  = W1;
                p2.Wt = W2;
                p2.b  = W3;
                p2.bt = W4;
                
            case 3
                p3.U  = W1;
                p3.bu = W2;
                
        end
        
        
        [f_xph] = compute_Fofx(X,Y,p1,p2,p3,f,nl,sl,cfn);
        
        switch argno
            case 1
                W1 = W_toCheck; W1(cr,cc) = W1(cr,cc) - h;
            case 2
                W2 = W_toCheck; W2(cr,cc) = W2(cr,cc) - h;
            case 3
                W3 = W_toCheck; W3(cr,cc) = W3(cr,cc) - h;
            case 4
                W4 = W_toCheck; W4(cr,cc) = W4(cr,cc) - h;
                
        end
        
        switch lno
            case 1
                p1.W  = W1;
                p1.Wt = W2;
                p1.b  = W3;
                p1.bt = W4;
                
            case 2
                p2.W  = W1;
                p2.Wt = W2;
                p2.b  = W3;
                p2.bt = W4;
                
            case 3
                p3.U  = W1;
                p3.bu = W2;
                
        end
        
        [f_xnh] = compute_Fofx(X,Y,p1,p2,p3,f,nl,sl,cfn);
        
        gWn(cr,cc) = (f_xph-f_xnh)/(2*h);
    end
end


end