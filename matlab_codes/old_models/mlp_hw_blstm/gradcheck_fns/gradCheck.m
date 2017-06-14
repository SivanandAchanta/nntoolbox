
lno = 1;
argno = 1; [gUn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_f_1.U,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbun]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_f_1.bu,argno,lno,dm_f_1,dm_f_2);

[dgUn] = compute_gradDiff_fn(gUn,p_f_1.gU,'gU1');
[dgbun] = compute_gradDiff_fn(gbun,p_f_1.gbu,'gbu1');

lno = lno + 1;
argno = 1; [gUn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_f_2.U,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbun]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_f_2.bu,argno,lno,dm_f_1,dm_f_2);

[dgUn] = compute_gradDiff_fn(gUn,p_f_2.gU,'gU2');
[dgbun] = compute_gradDiff_fn(gbun,p_f_2.gbu,'gbu2');


lno = lno + 1;
argno = 1; [gWn1] = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_1.W,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gWtn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_1.Wt,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_1.b,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbtn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_1.bt,argno,lno,dm_f_1,dm_f_2);

[dgWn1] = compute_gradDiff_fn(gWn1,p_h_1.gW,'gW1');
[dgWtn1] = compute_gradDiff_fn(gWtn1,p_h_1.gWt,'gWt1');
[dgbn1] = compute_gradDiff_fn(gbn1,p_h_1.gb,'gb1');
[dgbtn1] = compute_gradDiff_fn(gbtn1,p_h_1.gbt,'gbt1');


lno = lno + 1;
argno = 1; [gWn1] = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_2.W,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gWtn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_2.Wt,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_2.b,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbtn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_2.bt,argno,lno,dm_f_1,dm_f_2);

[dgWn1] = compute_gradDiff_fn(gWn1,p_h_2.gW,'gW2');
[dgWtn1] = compute_gradDiff_fn(gWtn1,p_h_2.gWt,'gWt2');
[dgbn1] = compute_gradDiff_fn(gbn1,p_h_2.gb,'gb2');
[dgbtn1] = compute_gradDiff_fn(gbtn1,p_h_2.gbt,'gbt2');

lno = lno + 1;
argno = 1; [gWn1] = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_3.W,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gWtn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_3.Wt,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_3.b,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbtn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_3.bt,argno,lno,dm_f_1,dm_f_2);

[dgWn1] = compute_gradDiff_fn(gWn1,p_h_3.gW,'gW3');
[dgWtn1] = compute_gradDiff_fn(gWtn1,p_h_3.gWt,'gWt3');
[dgbn1] = compute_gradDiff_fn(gbn1,p_h_3.gb,'gb3');
[dgbtn1] = compute_gradDiff_fn(gbtn1,p_h_3.gbt,'gbt3');

lno = lno + 1;
argno = 1; [gWn1] = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_4.W,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gWtn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_4.Wt,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_4.b,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbtn1]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_h_4.bt,argno,lno,dm_f_1,dm_f_2);

[dgWn1] = compute_gradDiff_fn(gWn1,p_h_4.gW,'gW4');
[dgWtn1] = compute_gradDiff_fn(gWtn1,p_h_4.gWt,'gWt4');
[dgbn1] = compute_gradDiff_fn(gbn1,p_h_4.gb,'gb4');
[dgbtn1] = compute_gradDiff_fn(gbtn1,p_h_4.gbt,'gbt4');

% lno = lno + 3;

lno = lno + 1;
argno = 1; [gWzn] = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.Wz,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gWin]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.Wi,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gWfn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.Wf,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gWon]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.Wo,argno,lno,dm_f_1,dm_f_2);

argno = argno + 1; [gRzn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.Rz,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gRin]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.Ri,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gRfn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.Rf,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gRon]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.Ro,argno,lno,dm_f_1,dm_f_2);

argno = argno + 1; [gbzn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.bz,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbin]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.bi,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbfn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.bf,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbon]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.bo,argno,lno,dm_f_1,dm_f_2);

argno = argno + 1; [gpin]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.pi,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gpfn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.pf,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gpon]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lf_1.po,argno,lno,dm_f_1,dm_f_2);

[dgWzn] = compute_gradDiff_fn(gWzn,p_lf_1.gWz,'gWzf');
[dgWin] = compute_gradDiff_fn(gWin,p_lf_1.gWi,'gWif');
[dgWfn] = compute_gradDiff_fn(gWfn,p_lf_1.gWf,'gWff');
[dgWon] = compute_gradDiff_fn(gWon,p_lf_1.gWo,'gWof');

[dgRzn] = compute_gradDiff_fn(gRzn,p_lf_1.gRz,'gRzf');
[dgRin] = compute_gradDiff_fn(gRin,p_lf_1.gRi,'gRif');
[dgRfn] = compute_gradDiff_fn(gRfn,p_lf_1.gRf,'gRff');
[dgRon] = compute_gradDiff_fn(gRon,p_lf_1.gRo,'gRof');

[dgbzn] = compute_gradDiff_fn(gbzn,p_lf_1.gbz,'gbzf');
[dgbin] = compute_gradDiff_fn(gbin,p_lf_1.gbi,'gbif');
[dgbfn] = compute_gradDiff_fn(gbfn,p_lf_1.gbf,'gbff');
[dgbon] = compute_gradDiff_fn(gbon,p_lf_1.gbo,'gbof');

[dgpin] = compute_gradDiff_fn(gpin,p_lf_1.gpi,'gpif');
[dgpfn] = compute_gradDiff_fn(gpfn,p_lf_1.gpf,'gpff');
[dgpon] = compute_gradDiff_fn(gpon,p_lf_1.gpo,'gpof');


lno = lno + 1;
argno = 1; [gWzn] = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.Wz,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gWin]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.Wi,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gWfn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.Wf,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gWon]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.Wo,argno,lno,dm_f_1,dm_f_2);

argno = argno + 1; [gRzn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.Rz,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gRin]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.Ri,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gRfn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.Rf,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gRon]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.Ro,argno,lno,dm_f_1,dm_f_2);

argno = argno + 1; [gbzn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.bz,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbin]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.bi,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbfn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.bf,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbon]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.bo,argno,lno,dm_f_1,dm_f_2);

argno = argno + 1; [gpin]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.pi,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gpfn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.pf,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gpon]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_lb_1.po,argno,lno,dm_f_1,dm_f_2);

[dgWzn] = compute_gradDiff_fn(gWzn,p_lb_1.gWz,'gWzb');
[dgWin] = compute_gradDiff_fn(gWin,p_lb_1.gWi,'gWib');
[dgWfn] = compute_gradDiff_fn(gWfn,p_lb_1.gWf,'gWfb');
[dgWon] = compute_gradDiff_fn(gWon,p_lb_1.gWo,'gWob');

[dgRzn] = compute_gradDiff_fn(gRzn,p_lb_1.gRz,'gRzb');
[dgRin] = compute_gradDiff_fn(gRin,p_lb_1.gRi,'gRib');
[dgRfn] = compute_gradDiff_fn(gRfn,p_lb_1.gRf,'gRfb');
[dgRon] = compute_gradDiff_fn(gRon,p_lb_1.gRo,'gRob');

[dgbzn] = compute_gradDiff_fn(gbzn,p_lb_1.gbz,'gbzb');
[dgbin] = compute_gradDiff_fn(gbin,p_lb_1.gbi,'gbib');
[dgbfn] = compute_gradDiff_fn(gbfn,p_lb_1.gbf,'gbfb');
[dgbon] = compute_gradDiff_fn(gbon,p_lb_1.gbo,'gbob');

[dgpin] = compute_gradDiff_fn(gpin,p_lb_1.gpi,'gpib');
[dgpfn] = compute_gradDiff_fn(gpfn,p_lb_1.gpf,'gpfb');
[dgpon] = compute_gradDiff_fn(gpon,p_lb_1.gpo,'gpob');


lno = lno + 1;
argno = 1; [gUn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_f_3_1.U,argno,lno,dm_f_1,dm_f_2);
argno = argno + 1; [gbun]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_f_3_1.bu,argno,lno,dm_f_1,dm_f_2);

[dgUn] = compute_gradDiff_fn(gUn,p_f_3_1.gU,'gU3');
[dgbun] = compute_gradDiff_fn(gbun,p_f_3_1.gbu,'gbu3');

lno = lno + 1;
argno = 1; [gUn]  = compute_NumericalGrad(p_f_1,p_f_2,p_h_1,p_h_2,p_h_3,p_h_4,p_lf_1,p_lb_1,p_f_3_1,p_f_3_2,X,Y,f,nl,sl,cfn,p_f_3_2.U,argno,lno,dm_f_1,dm_f_2);

[dgUn] = compute_gradDiff_fn(gUn,p_f_3_2.gU,'gU4');

pause
