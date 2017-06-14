function [gW] = compute_NumericalGrad(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,X,Y,f,nl,sl,cfn,W_toCheck,argno,lno,dm1,dm2)

switch lno
    case 1
        W1 = p1.U;
        W2 = p1.bu;
        
    case 2
        W1 = p2.U;
        W2 = p2.bu;
        
    case 3
        W1 = p3.W;
        W2 = p3.Wt;
        W3 = p3.b;
        W4 = p3.bt;
        
    case 4
        W1 = p4.W;
        W2 = p4.Wt;
        W3 = p4.b;
        W4 = p4.bt;
        
    case 5
        W1 = p5.W;
        W2 = p5.Wt;
        W3 = p5.b;
        W4 = p5.bt;
        
    case 6
        W1 = p6.W;
        W2 = p6.Wt;
        W3 = p6.b;
        W4 = p6.bt;
        
    case 7
        W1 = p7.Wz;
        W2 = p7.Wi;
        W3 = p7.Wf;
        W4 = p7.Wo;
        
        W5 = p7.Rz;
        W6 = p7.Ri;
        W7 = p7.Rf;
        W8 = p7.Ro;
        
        W9 = p7.bz;
        W10 = p7.bi;
        W11 = p7.bf;
        W12 = p7.bo;
        
        W13 = p7.pi;
        W14 = p7.pf;
        W15 = p7.po;
        
    case 8
        
        W1 = p8.Wz;
        W2 = p8.Wi;
        W3 = p8.Wf;
        W4 = p8.Wo;
        
        W5 = p8.Rz;
        W6 = p8.Ri;
        W7 = p8.Rf;
        W8 = p8.Ro;
        
        W9 = p8.bz;
        W10 = p8.bi;
        W11 = p8.bf;
        W12 = p8.bo;
        
        W13 = p8.pi;
        W14 = p8.pf;
        W15 = p8.po;
        
    case 9
        W1 = p9.U;
        W2 = p9.bu;
        
    case 10
        W1 = p10.U;
        
end

% perturbation magnitude at x
h = 1e-3;

% Compute the numerical gradient
[nr,nc]  = size(W_toCheck);
gW       = zeros(nr,nc);

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
            case 5
                W5 = W_toCheck; W5(cr,cc) = W5(cr,cc) + h;
            case 6
                W6 = W_toCheck; W6(cr,cc) = W6(cr,cc) + h;
            case 7
                W7 = W_toCheck; W7(cr,cc) = W7(cr,cc) + h;
            case 8
                W8 = W_toCheck; W8(cr,cc) = W8(cr,cc) + h;
            case 9
                W9 = W_toCheck; W9(cr,cc) = W9(cr,cc) + h;
            case 10
                W10 = W_toCheck; W10(cr,cc) = W10(cr,cc) + h;
            case 11
                W11 = W_toCheck; W11(cr,cc) = W11(cr,cc) + h;
            case 12
                W12 = W_toCheck; W12(cr,cc) = W12(cr,cc) + h;
            case 13
                W13 = W_toCheck; W13(cr,cc) = W13(cr,cc) + h;
            case 14
                W14 = W_toCheck; W14(cr,cc) = W14(cr,cc) + h;
            case 15
                W15 = W_toCheck; W15(cr,cc) = W15(cr,cc) + h;
                
        end
        
        switch lno
            case 1
                p1.U  = W1;
                p1.bu = W2;
                
            case 2
                p2.U  = W1;
                p2.bu = W2;
                
            case 3
                p3.W  = W1;
                p3.Wt = W2;
                p3.b  = W3;
                p3.bt = W4;
                
            case 4
                p4.W  = W1;
                p4.Wt = W2;
                p4.b  = W3;
                p4.bt = W4;
                
            case 5
                p5.W  = W1;
                p5.Wt = W2;
                p5.b  = W3;
                p5.bt = W4;
                
            case 6
                p6.W  = W1;
                p6.Wt = W2;
                p6.b  = W3;
                p6.bt = W4;
                
            case 7
                p7.Wz = W1;
                p7.Wi = W2;
                p7.Wf = W3;
                p7.Wo = W4;
                
                p7.Rz = W5;
                p7.Ri = W6;
                p7.Rf = W7;
                p7.Ro = W8;
                
                p7.bz = W9;
                p7.bi = W10;
                p7.bf = W11;
                p7.bo = W12;
                
                p7.pi = W13;
                p7.pf = W14;
                p7.po = W15;
                
            case 8
                p8.Wz = W1;
                p8.Wi = W2;
                p8.Wf = W3;
                p8.Wo = W4;
                
                p8.Rz = W5;
                p8.Ri = W6;
                p8.Rf = W7;
                p8.Ro = W8;
                
                p8.bz = W9;
                p8.bi = W10;
                p8.bf = W11;
                p8.bo = W12;
                
                p8.pi = W13;
                p8.pf = W14;
                p8.po = W15;
                
            case 9
                p9.U  = W1;
                p9.bu = W2;
                
            case 10
                p10.U  = W1;
                
        end
        
        [f_xph] = compute_Fofx(X,Y,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,nl,sl,f,cfn,dm1,dm2);
        
        
        switch argno
            case 1
                W1 = W_toCheck; W1(cr,cc) = W1(cr,cc) - h;
            case 2
                W2 = W_toCheck; W2(cr,cc) = W2(cr,cc) - h;
            case 3
                W3 = W_toCheck; W3(cr,cc) = W3(cr,cc) - h;
            case 4
                W4 = W_toCheck; W4(cr,cc) = W4(cr,cc) - h;
            case 5
                W5 = W_toCheck; W5(cr,cc) = W5(cr,cc)- h;
            case 6
                W6 = W_toCheck; W6(cr,cc) = W6(cr,cc)- h;
            case 7
                W7 = W_toCheck; W7(cr,cc) = W7(cr,cc)- h;
            case 8
                W8 = W_toCheck; W8(cr,cc) = W8(cr,cc)- h;
            case 9
                W9 = W_toCheck; W9(cr,cc) = W9(cr,cc)- h;
            case 10
                W10 = W_toCheck; W10(cr,cc) = W10(cr,cc)- h;
            case 11
                W11 = W_toCheck; W11(cr,cc) = W11(cr,cc)- h;
            case 12
                W12 = W_toCheck; W12(cr,cc) = W12(cr,cc)- h;
            case 13
                W13 = W_toCheck; W13(cr,cc) = W13(cr,cc)- h;
            case 14
                W14 = W_toCheck; W14(cr,cc) = W14(cr,cc)- h;
            case 15
                W15 = W_toCheck; W15(cr,cc) = W15(cr,cc)- h;
        end
        
        switch lno
            case 1
                p1.U  = W1;
                p1.bu = W2;
                
            case 2
                p2.U  = W1;
                p2.bu = W2;
                
            case 3
                p3.W  = W1;
                p3.Wt = W2;
                p3.b  = W3;
                p3.bt = W4;
                
            case 4
                p4.W  = W1;
                p4.Wt = W2;
                p4.b  = W3;
                p4.bt = W4;
                
            case 5
                p5.W  = W1;
                p5.Wt = W2;
                p5.b  = W3;
                p5.bt = W4;
                
            case 6
                p6.W  = W1;
                p6.Wt = W2;
                p6.b  = W3;
                p6.bt = W4;
                
            case 7
                p7.Wz = W1;
                p7.Wi = W2;
                p7.Wf = W3;
                p7.Wo = W4;
                
                p7.Rz = W5;
                p7.Ri = W6;
                p7.Rf = W7;
                p7.Ro = W8;
                
                p7.bz = W9;
                p7.bi = W10;
                p7.bf = W11;
                p7.bo = W12;
                
                p7.pi = W13;
                p7.pf = W14;
                p7.po = W15;
                
            case 8
                p8.Wz = W1;
                p8.Wi = W2;
                p8.Wf = W3;
                p8.Wo = W4;
                
                p8.Rz = W5;
                p8.Ri = W6;
                p8.Rf = W7;
                p8.Ro = W8;
                
                p8.bz = W9;
                p8.bi = W10;
                p8.bf = W11;
                p8.bo = W12;
                
                p8.pi = W13;
                p8.pf = W14;
                p8.po = W15;
                
            case 9
                p9.U  = W1;
                p9.bu = W2;
                
            case 10
                p10.U  = W1;
                
        end
        
        
        [f_xnh] = compute_Fofx(X,Y,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,nl,sl,f,cfn,dm1,dm2);
        
        gW(cr,cc) = (f_xph-f_xnh)/(2*h);
    end
end


end