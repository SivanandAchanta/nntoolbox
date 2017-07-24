function [p,Ebb] = bp_clb(p,Eb,ym,X,K,C,nin,nout,f,sl)

K = 1; [p.gU1,o1] = bp_cl_v2(p.U1,Eb(:,1:C),ym(:,1:C),X,K,nin,C,f,sl);
K = 2; [p.gU2,o2] = bp_cl_v2(p.U2,Eb(:,C + 1:2*C),ym(:,C + 1:2*C),X,K,nin,C,f,sl);
K = 3; [p.gU3,o3] = bp_cl_v2(p.U3,Eb(:,2*C + 1:3*C),ym(:,2*C + 1:3*C),X,K,nin,C,f,sl);
K = 4; [p.gU4,o4] = bp_cl_v2(p.U4,Eb(:,3*C + 1:4*C),ym(:,3*C + 1:4*C),X,K,nin,C,f,sl);
K = 5; [p.gU5,o5] = bp_cl_v2(p.U5,Eb(:,4*C + 1:5*C),ym(:,4*C + 1:5*C),X,K,nin,C,f,sl);
K = 6; [p.gU6,o6] = bp_cl_v2(p.U6,Eb(:,5*C + 1:6*C),ym(:,5*C + 1:6*C),X,K,nin,C,f,sl);
K = 7; [p.gU7,o7] = bp_cl_v2(p.U7,Eb(:,6*C + 1:7*C),ym(:,6*C + 1:7*C),X,K,nin,C,f,sl);
K = 8; [p.gU8,o8] = bp_cl_v2(p.U8,Eb(:,7*C + 1:8*C),ym(:,7*C + 1:8*C),X,K,nin,C,f,sl);


Ebb = o1 + o2 + o3 + o4 + o5 + o6 + o7 + o8;


end