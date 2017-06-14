function [gW] = compute_NumericalGrad(p1,p2,p3,p4,p5,p6,X,Y,f,nl,sl,cfn,W_toCheck,argno,lno)

switch lno
    case 1
        W1 = p1.U;
        W2 = p1.bu;
        
    case 2
        W1 = p2.Wz;
        W2 = p2.Wi;
        W3 = p2.Wf;
        W4 = p2.Wo;
        
        W5 = p2.Rz;
        W6 = p2.Ri;
        W7 = p2.Rf;
        W8 = p2.Ro;
        
        W9 = p2.bz;
        W10 = p2.bi;
        W11 = p2.bf;
        W12 = p2.bo;
        
        W13 = p2.pi;
        W14 = p2.pf;
        W15 = p2.po;
        
    case 3
        
        W1 = p3.Wz;
        W2 = p3.Wi;
        W3 = p3.Wf;
        W4 = p3.Wo;
        
        W5 = p3.Rz;
        W6 = p3.Ri;
        W7 = p3.Rf;
        W8 = p3.Ro;
        
        W9 = p3.bz;
        W10 = p3.bi;
        W11 = p3.bf;
        W12 = p3.bo;
        
        W13 = p3.pi;
        W14 = p3.pf;
        W15 = p3.po;
        
    case 4
        W1 = p4.Wz;
        W2 = p4.Wi;
        W3 = p4.Wf;
        W4 = p4.Wo;
        
        W5 = p4.Rz;
        W6 = p4.Ri;
        W7 = p4.Rf;
        W8 = p4.Ro;
        
        W9 = p4.bz;
        W10 = p4.bi;
        W11 = p4.bf;
        W12 = p4.bo;
        
        W13 = p4.pi;
        W14 = p4.pf;
        W15 = p4.po;
        
    case 5
        
        W1 = p5.Wz;
        W2 = p5.Wi;
        W3 = p5.Wf;
        W4 = p5.Wo;
        
        W5 = p5.Rz;
        W6 = p5.Ri;
        W7 = p5.Rf;
        W8 = p5.Ro;
        
        W9 = p5.bz;
        W10 = p5.bi;
        W11 = p5.bf;
        W12 = p5.bo;
        
        W13 = p5.pi;
        W14 = p5.pf;
        W15 = p5.po;
        
    case 6
        W1 = p6.U;
        W2 = p6.bu;
        
end

% perturbation magnitude at x
h = 1e-5;

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
                p2.Wz = W1;
                p2.Wi = W2;
                p2.Wf = W3;
                p2.Wo = W4;
                
                p2.Rz = W5;
                p2.Ri = W6;
                p2.Rf = W7;
                p2.Ro = W8;
                
                p2.bz = W9;
                p2.bi = W10;
                p2.bf = W11;
                p2.bo = W12;
                
                p2.pi = W13;
                p2.pf = W14;
                p2.po = W15;
                
            case 3
                p3.Wz = W1;
                p3.Wi = W2;
                p3.Wf = W3;
                p3.Wo = W4;
                
                p3.Rz = W5;
                p3.Ri = W6;
                p3.Rf = W7;
                p3.Ro = W8;
                
                p3.bz = W9;
                p3.bi = W10;
                p3.bf = W11;
                p3.bo = W12;
                
                p3.pi = W13;
                p3.pf = W14;
                p3.po = W15;
                
            case 4
                p4.Wz = W1;
                p4.Wi = W2;
                p4.Wf = W3;
                p4.Wo = W4;
                
                p4.Rz = W5;
                p4.Ri = W6;
                p4.Rf = W7;
                p4.Ro = W8;
                
                p4.bz = W9;
                p4.bi = W10;
                p4.bf = W11;
                p4.bo = W12;
                
                p4.pi = W13;
                p4.pf = W14;
                p4.po = W15;
                
            case 5
                p5.Wz = W1;
                p5.Wi = W2;
                p5.Wf = W3;
                p5.Wo = W4;
                
                p5.Rz = W5;
                p5.Ri = W6;
                p5.Rf = W7;
                p5.Ro = W8;
                
                p5.bz = W9;
                p5.bi = W10;
                p5.bf = W11;
                p5.bo = W12;
                
                p5.pi = W13;
                p5.pf = W14;
                p5.po = W15;
                
            case 6
                p6.U  = W1;
                p6.bu = W2;
                
        end
        
        [f_xph] = compute_Fofx(X,Y,p1,p2,p3,p4,p5,p6,nl,sl,f,cfn);
        
        
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
                p2.Wz = W1;
                p2.Wi = W2;
                p2.Wf = W3;
                p2.Wo = W4;
                
                p2.Rz = W5;
                p2.Ri = W6;
                p2.Rf = W7;
                p2.Ro = W8;
                
                p2.bz = W9;
                p2.bi = W10;
                p2.bf = W11;
                p2.bo = W12;
                
                p2.pi = W13;
                p2.pf = W14;
                p2.po = W15;
                
            case 3
                p3.Wz = W1;
                p3.Wi = W2;
                p3.Wf = W3;
                p3.Wo = W4;
                
                p3.Rz = W5;
                p3.Ri = W6;
                p3.Rf = W7;
                p3.Ro = W8;
                
                p3.bz = W9;
                p3.bi = W10;
                p3.bf = W11;
                p3.bo = W12;
                
                p3.pi = W13;
                p3.pf = W14;
                p3.po = W15;
                
            case 4
                p4.Wz = W1;
                p4.Wi = W2;
                p4.Wf = W3;
                p4.Wo = W4;
                
                p4.Rz = W5;
                p4.Ri = W6;
                p4.Rf = W7;
                p4.Ro = W8;
                
                p4.bz = W9;
                p4.bi = W10;
                p4.bf = W11;
                p4.bo = W12;
                
                p4.pi = W13;
                p4.pf = W14;
                p4.po = W15;
                
            case 5
                p5.Wz = W1;
                p5.Wi = W2;
                p5.Wf = W3;
                p5.Wo = W4;
                
                p5.Rz = W5;
                p5.Ri = W6;
                p5.Rf = W7;
                p5.Ro = W8;
                
                p5.bz = W9;
                p5.bi = W10;
                p5.bf = W11;
                p5.bo = W12;
                
                p5.pi = W13;
                p5.pf = W14;
                p5.po = W15;
                
                
                
            case 6
                p6.U  = W1;
                p6.bu = W2;
                
        end
        
        
        [f_xnh] = compute_Fofx(X,Y,p1,p2,p3,p4,p5,p6,nl,sl,f,cfn);
        
        gW(cr,cc) = (f_xph-f_xnh)/(2*h);
    end
end


end