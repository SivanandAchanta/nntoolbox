function [gW] = compute_NumericalGrad(p1,p2,p3,p4,X,Y,f,nl,sl,cfn,W_toCheck,argno,lno)

switch lno
    case 1
        W1 = p1.Wz;
        W2 = p1.Wi;
        W3 = p1.Wf;
        W4 = p1.Wo;
        
        W5 = p1.Rz;
        W6 = p1.Ri;
        W7 = p1.Rf;
        W8 = p1.Ro;
        
        W9 = p1.bz;
        W10 = p1.bi;
        W11 = p1.bf;
        W12 = p1.bo;
        
        W13 = p1.pi;
        W14 = p1.pf;
        W15 = p1.po;
        
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
        W1 = p3.U;

    case 4
        W1 = p4.U;
        W2 = p4.bu;
        
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
                p1.Wz = W1;
                p1.Wi = W2;
                p1.Wf = W3;
                p1.Wo = W4;

                p1.Rz = W5;
                p1.Ri = W6;
                p1.Rf = W7;
                p1.Ro = W8;

                p1.bz = W9;
                p1.bi = W10;
                p1.bf = W11;
                p1.bo = W12;

                p1.pi = W13;
                p1.pf = W14;
                p1.po = W15;

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
                p3.U  = W1;
                
            case 4
                p4.U  = W1;
                p4.bu = W2;
                
        end
        
        [f_xph] = compute_Fofx(X,Y,p1,p2,p3,p4,nl,sl,f,cfn);
        
        
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
                p1.Wz = W1;
                p1.Wi = W2;
                p1.Wf = W3;
                p1.Wo = W4;

                p1.Rz = W5;
                p1.Ri = W6;
                p1.Rf = W7;
                p1.Ro = W8;

                p1.bz = W9;
                p1.bi = W10;
                p1.bf = W11;
                p1.bo = W12;

                p1.pi = W13;
                p1.pf = W14;
                p1.po = W15;

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
                p3.U  = W1;

            case 4
                p4.U  = W1;
                p4.bu = W2;

        end
 
        [f_xnh] = compute_Fofx(X,Y,p1,p2,p3,p4,nl,sl,f,cfn);
        
        gW(cr,cc) = (f_xph-f_xnh)/(2*h);
    end
end


end
