% Error at Output Layer
get_oplayer_error

% Backprop top layer
p_f_2.gU = (E'*hbm1);
p_f_2.gbu = sum(E,1)';
p_f_1.gU = (E'*hfm1);

% Backprop thru BLSTM layer
Ebb = E*p_f_2.U;
Ebf = E*p_f_1.U;

[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_1,zfm1,ifm1,ffm1,cfm1,ofm1,hcfm1,nl(2),sl,'frnn');
[p_lf_1] = gradients_lstm(X,p_lf_1,hfm1,cfm1,dom,dfm,dim,dzm,nl(2),'frnn');

[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb',p_lb_1,zbm1,ibm1,fbm1,cbm1,obm1,hcbm1,nl(2),sl,'brnn');
[p_lb_1] = gradients_lstm(X,p_lb_1,hbm1,cbm1,dom,dfm,dim,dzm,nl(2),'brnn');

