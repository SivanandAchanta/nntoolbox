function [gW] = compute_NumericalGrad(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,X,Y,Yn,f,nl,sl_enc,sl_dec,cfn,W_toCheck,argno,lno,dm1,dm2,dm3,dm4)

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
        W1 = p9.bu;
        
    case 10
        W1 = p10.U;
        W2 = p10.bu;
        
    case 11
        W1 = p11.U;
        
    case 12
        W1 = p12.U;
        W2 = p12.bu;
        
    case 13
        W1 = p13.U;
        W2 = p13.bu;
        
    case 14
        W1 = p14.Wz;
        W2 = p14.Wi;
        W3 = p14.Wf;
        W4 = p14.Wo;
        
        W5 = p14.Rz;
        W6 = p14.Ri;
        W7 = p14.Rf;
        W8 = p14.Ro;
        
        W9 = p14.bz;
        W10 = p14.bi;
        W11 = p14.bf;
        W12 = p14.bo;
        
        W13 = p14.pi;
        W14 = p14.pf;
        W15 = p14.po;
        
        
    case 15
        W1 = p15.Wz;
        W2 = p15.Wi;
        W3 = p15.Wf;
        W4 = p15.Wo;
        
        W5 = p15.Rz;
        W6 = p15.Ri;
        W7 = p15.Rf;
        W8 = p15.Ro;
        
        W9 = p15.bz;
        W10 = p15.bi;
        W11 = p15.bf;
        W12 = p15.bo;
        
        W13 = p15.pi;
        W14 = p15.pf;
        W15 = p15.po;
        
        
    case 16
        W1 = p16.Wz;
        W2 = p16.Wi;
        W3 = p16.Wf;
        W4 = p16.Wo;
        
        W5 = p16.Rz;
        W6 = p16.Ri;
        W7 = p16.Rf;
        W8 = p16.Ro;
        
        W9 = p16.bz;
        W10 = p16.bi;
        W11 = p16.bf;
        W12 = p16.bo;
        
        W13 = p16.pi;
        W14 = p16.pf;
        W15 = p16.po;
        
    case 17
        W1 = p17.U;
        W2 = p17.bu;
        
    case 18
        W1 = p18.U;
        W2 = p18.bu;
        
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
                p9.bu = W1;
                
            case 10
                p10.U  = W1;
                p10.bu = W2;
                
            case 11
                p11.U  = W1;
                
            case 12
                p12.U  = W1;
                p12.bu = W2;
                
            case 13
                p13.U  = W1;
                p13.bu = W2;
                
            case 14
                p14.Wz = W1;
                p14.Wi = W2;
                p14.Wf = W3;
                p14.Wo = W4;
                
                p14.Rz = W5;
                p14.Ri = W6;
                p14.Rf = W7;
                p14.Ro = W8;
                
                p14.bz = W9;
                p14.bi = W10;
                p14.bf = W11;
                p14.bo = W12;
                
                p14.pi = W13;
                p14.pf = W14;
                p14.po = W15;
                
            case 15
                p15.Wz = W1;
                p15.Wi = W2;
                p15.Wf = W3;
                p15.Wo = W4;
                
                p15.Rz = W5;
                p15.Ri = W6;
                p15.Rf = W7;
                p15.Ro = W8;
                
                p15.bz = W9;
                p15.bi = W10;
                p15.bf = W11;
                p15.bo = W12;
                
                p15.pi = W13;
                p15.pf = W14;
                p15.po = W15;
                
            case 16
                p16.Wz = W1;
                p16.Wi = W2;
                p16.Wf = W3;
                p16.Wo = W4;
                
                p16.Rz = W5;
                p16.Ri = W6;
                p16.Rf = W7;
                p16.Ro = W8;
                
                p16.bz = W9;
                p16.bi = W10;
                p16.bf = W11;
                p16.bo = W12;
                
                p16.pi = W13;
                p16.pf = W14;
                p16.po = W15;
                
                
            case 17
                p17.U  = W1;
                p17.bu = W2;
                
            case 18
                p18.U  = W1;
                p18.bu = W2;
                
        end
        
        [f_xph] = compute_Fofx(X,Y,Yn,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,nl,sl_enc,sl_dec,f,cfn,dm1,dm2,dm3,dm4);
        
        
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
                p9.bu = W1;
                
            case 10
                p10.U  = W1;
                p10.bu = W2;
                
            case 11
                p11.U  = W1;
                
            case 12
                p12.U  = W1;
                p12.bu = W2;
                
            case 13
                p13.U  = W1;
                p13.bu = W2;
                
            case 14
                p14.Wz = W1;
                p14.Wi = W2;
                p14.Wf = W3;
                p14.Wo = W4;
                
                p14.Rz = W5;
                p14.Ri = W6;
                p14.Rf = W7;
                p14.Ro = W8;
                
                p14.bz = W9;
                p14.bi = W10;
                p14.bf = W11;
                p14.bo = W12;
                
                p14.pi = W13;
                p14.pf = W14;
                p14.po = W15;
                
            case 15
                p15.Wz = W1;
                p15.Wi = W2;
                p15.Wf = W3;
                p15.Wo = W4;
                
                p15.Rz = W5;
                p15.Ri = W6;
                p15.Rf = W7;
                p15.Ro = W8;
                
                p15.bz = W9;
                p15.bi = W10;
                p15.bf = W11;
                p15.bo = W12;
                
                p15.pi = W13;
                p15.pf = W14;
                p15.po = W15;
                
            case 16
                p16.Wz = W1;
                p16.Wi = W2;
                p16.Wf = W3;
                p16.Wo = W4;
                
                p16.Rz = W5;
                p16.Ri = W6;
                p16.Rf = W7;
                p16.Ro = W8;
                
                p16.bz = W9;
                p16.bi = W10;
                p16.bf = W11;
                p16.bo = W12;
                
                p16.pi = W13;
                p16.pf = W14;
                p16.po = W15;
                
                
            case 17
                p17.U  = W1;
                p17.bu = W2;
                
            case 18
                p18.U  = W1;
                p18.bu = W2;
                
        end
        
        
        [f_xnh] = compute_Fofx(X,Y,Yn,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,nl,sl_enc,sl_dec,f,cfn,dm1,dm2,dm3,dm4);
        
        gW(cr,cc) = (f_xph-f_xnh)/(2*h);
    end
end


end