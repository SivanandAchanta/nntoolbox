lno = 1;
argno = 1; [gUn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_f_1.U,argno,lno);
argno = argno + 1; [gbun]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_f_1.bu,argno,lno);

[dgUn] = compute_gradDiff_fn(gUn,p_f_1.gU,'gU1');
[dgbun] = compute_gradDiff_fn(gbun,p_f_1.gbu,'gbu1');

lno = lno + 1;
argno = 1; [gWzn] = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.Wz,argno,lno);
argno = argno + 1; [gWin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.Wi,argno,lno);
argno = argno + 1; [gWfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.Wf,argno,lno);
argno = argno + 1; [gWon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.Wo,argno,lno);

argno = argno + 1; [gRzn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.Rz,argno,lno);
argno = argno + 1; [gRin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.Ri,argno,lno);
argno = argno + 1; [gRfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.Rf,argno,lno);
argno = argno + 1; [gRon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.Ro,argno,lno);

argno = argno + 1; [gbzn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.bz,argno,lno);
argno = argno + 1; [gbin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.bi,argno,lno);
argno = argno + 1; [gbfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.bf,argno,lno);
argno = argno + 1; [gbon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.bo,argno,lno);

argno = argno + 1; [gpin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.pi,argno,lno);
argno = argno + 1; [gpfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.pf,argno,lno);
argno = argno + 1; [gpon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_1.po,argno,lno);

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
argno = 1; [gWzn] = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.Wz,argno,lno);
argno = argno + 1; [gWin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.Wi,argno,lno);
argno = argno + 1; [gWfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.Wf,argno,lno);
argno = argno + 1; [gWon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.Wo,argno,lno);

argno = argno + 1; [gRzn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.Rz,argno,lno);
argno = argno + 1; [gRin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.Ri,argno,lno);
argno = argno + 1; [gRfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.Rf,argno,lno);
argno = argno + 1; [gRon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.Ro,argno,lno);

argno = argno + 1; [gbzn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.bz,argno,lno);
argno = argno + 1; [gbin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.bi,argno,lno);
argno = argno + 1; [gbfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.bf,argno,lno);
argno = argno + 1; [gbon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.bo,argno,lno);

argno = argno + 1; [gpin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.pi,argno,lno);
argno = argno + 1; [gpfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.pf,argno,lno);
argno = argno + 1; [gpon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_1.po,argno,lno);

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
argno = 1; [gWzn] = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.Wz,argno,lno);
argno = argno + 1; [gWin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.Wi,argno,lno);
argno = argno + 1; [gWfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.Wf,argno,lno);
argno = argno + 1; [gWon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.Wo,argno,lno);

argno = argno + 1; [gRzn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.Rz,argno,lno);
argno = argno + 1; [gRin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.Ri,argno,lno);
argno = argno + 1; [gRfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.Rf,argno,lno);
argno = argno + 1; [gRon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.Ro,argno,lno);

argno = argno + 1; [gbzn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.bz,argno,lno);
argno = argno + 1; [gbin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.bi,argno,lno);
argno = argno + 1; [gbfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.bf,argno,lno);
argno = argno + 1; [gbon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.bo,argno,lno);

argno = argno + 1; [gpin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.pi,argno,lno);
argno = argno + 1; [gpfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.pf,argno,lno);
argno = argno + 1; [gpon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lf_2.po,argno,lno);

[dgWzn] = compute_gradDiff_fn(gWzn,p_lf_2.gWz,'gWzf');
[dgWin] = compute_gradDiff_fn(gWin,p_lf_2.gWi,'gWif');
[dgWfn] = compute_gradDiff_fn(gWfn,p_lf_2.gWf,'gWff');
[dgWon] = compute_gradDiff_fn(gWon,p_lf_2.gWo,'gWof');

[dgRzn] = compute_gradDiff_fn(gRzn,p_lf_2.gRz,'gRzf');
[dgRin] = compute_gradDiff_fn(gRin,p_lf_2.gRi,'gRif');
[dgRfn] = compute_gradDiff_fn(gRfn,p_lf_2.gRf,'gRff');
[dgRon] = compute_gradDiff_fn(gRon,p_lf_2.gRo,'gRof');

[dgbzn] = compute_gradDiff_fn(gbzn,p_lf_2.gbz,'gbzf');
[dgbin] = compute_gradDiff_fn(gbin,p_lf_2.gbi,'gbif');
[dgbfn] = compute_gradDiff_fn(gbfn,p_lf_2.gbf,'gbff');
[dgbon] = compute_gradDiff_fn(gbon,p_lf_2.gbo,'gbof');

[dgpin] = compute_gradDiff_fn(gpin,p_lf_2.gpi,'gpif');
[dgpfn] = compute_gradDiff_fn(gpfn,p_lf_2.gpf,'gpff');
[dgpon] = compute_gradDiff_fn(gpon,p_lf_2.gpo,'gpof');


lno = lno + 1;
argno = 1; [gWzn] = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.Wz,argno,lno);
argno = argno + 1; [gWin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.Wi,argno,lno);
argno = argno + 1; [gWfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.Wf,argno,lno);
argno = argno + 1; [gWon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.Wo,argno,lno);

argno = argno + 1; [gRzn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.Rz,argno,lno);
argno = argno + 1; [gRin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.Ri,argno,lno);
argno = argno + 1; [gRfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.Rf,argno,lno);
argno = argno + 1; [gRon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.Ro,argno,lno);

argno = argno + 1; [gbzn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.bz,argno,lno);
argno = argno + 1; [gbin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.bi,argno,lno);
argno = argno + 1; [gbfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.bf,argno,lno);
argno = argno + 1; [gbon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.bo,argno,lno);

argno = argno + 1; [gpin]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.pi,argno,lno);
argno = argno + 1; [gpfn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.pf,argno,lno);
argno = argno + 1; [gpon]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_lb_2.po,argno,lno);

[dgWzn] = compute_gradDiff_fn(gWzn,p_lb_2.gWz,'gWzb');
[dgWin] = compute_gradDiff_fn(gWin,p_lb_2.gWi,'gWib');
[dgWfn] = compute_gradDiff_fn(gWfn,p_lb_2.gWf,'gWfb');
[dgWon] = compute_gradDiff_fn(gWon,p_lb_2.gWo,'gWob');

[dgRzn] = compute_gradDiff_fn(gRzn,p_lb_2.gRz,'gRzb');
[dgRin] = compute_gradDiff_fn(gRin,p_lb_2.gRi,'gRib');
[dgRfn] = compute_gradDiff_fn(gRfn,p_lb_2.gRf,'gRfb');
[dgRon] = compute_gradDiff_fn(gRon,p_lb_2.gRo,'gRob');

[dgbzn] = compute_gradDiff_fn(gbzn,p_lb_2.gbz,'gbzb');
[dgbin] = compute_gradDiff_fn(gbin,p_lb_2.gbi,'gbib');
[dgbfn] = compute_gradDiff_fn(gbfn,p_lb_2.gbf,'gbfb');
[dgbon] = compute_gradDiff_fn(gbon,p_lb_2.gbo,'gbob');

[dgpin] = compute_gradDiff_fn(gpin,p_lb_2.gpi,'gpib');
[dgpfn] = compute_gradDiff_fn(gpfn,p_lb_2.gpf,'gpfb');
[dgpon] = compute_gradDiff_fn(gpon,p_lb_2.gpo,'gpob');


lno = lno + 1;
argno = 1; [gUn]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_f_2.U,argno,lno);
argno = argno + 1; [gbun]  = compute_NumericalGrad(p_f_1,p_lf_1,p_lb_1,p_lf_2,p_lb_2,p_f_2,X,Y,f,nl,sl,cfn,p_f_2.bu,argno,lno);

[dgUn] = compute_gradDiff_fn(gUn,p_f_2.gU,'gU2');
[dgbun] = compute_gradDiff_fn(gbun,p_f_2.gbu,'gbu2');

