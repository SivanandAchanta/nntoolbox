% Network 2

% Error at Output Layer
E = get_oplayer_error(Y2,ym_2,2*sl_dec,cfn);

% Backprop top ms
p_f_3_1_N2.gU = (E'*hfm);
p_f_3_2_N2.gU = (E'*hbm);
p_f_3_1_N2.gbu = sum(E,1)';

Ebf = p_f_3_1_N2.U'*E';
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf,p_lf_1_N2,zfm,ifm,ffm,cfm,ofm,hcfm,nl(9),2*sl_dec,'frnn');
[p_lf_1_N2] = gradients_lstm(hm_h_4_N2,p_lf_1_N2,hfm,cfm,dom,dfm,dim,dzm,nl(9),'frnn');

Ebb = p_f_3_2_N2.U'*E';
[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb,p_lb_1_N2,zbm,ibm,fbm,cbm,obm,hcbm,nl(9),2*sl_dec,'brnn');
[p_lb_1_N2] = gradients_lstm(hm_h_4_N2,p_lb_1_N2,hbm,cbm,dom,dfm,dim,dzm,nl(9),'brnn');

Eb = Ebf + Ebb;
% Eb = Ebf;

[p_h_4_N2.gW,p_h_4_N2.gb,p_h_4_N2.gWt,p_h_4_N2.gbt,Eb] = bp_hl(nl(8),2*sl_dec,Eb,p_h_4_N2,htm_h_4_N2,tm_h_4_N2,hm_h_3_N2,f(7));
[p_h_3_N2.gW,p_h_3_N2.gb,p_h_3_N2.gWt,p_h_3_N2.gbt,Eb] = bp_hl(nl(8),2*sl_dec,Eb,p_h_3_N2,htm_h_3_N2,tm_h_3_N2,hm_h_2_N2,f(7));
[p_h_2_N2.gW,p_h_2_N2.gb,p_h_2_N2.gWt,p_h_2_N2.gbt,Eb] = bp_hl(nl(8),2*sl_dec,Eb,p_h_2_N2,htm_h_2_N2,tm_h_2_N2,hm_h_1_N2,f(7));
[p_h_1_N2.gW,p_h_1_N2.gb,p_h_1_N2.gWt,p_h_1_N2.gbt,Eb] = bp_hl(nl(8),2*sl_dec,Eb,p_h_1_N2,htm_h_1_N2,tm_h_1_N2,hm_c_4_N2,f(7));

% convulution layers with residual connection
Ebp = Eb;
[p_c_3_N2,Eb] = bp_cl(p_c_3_N2,Eb,hm_c_3_N2,hm_c_2_N2,K_conv_l2,C_conv_l2,C_conv_l2,'L',2*sl_dec);
[p_c_2_N2,Eb] = bp_cl(p_c_2_N2,Eb,hm_c_2_N2,hm_c_1_N2,K_conv_l2,K_conv_l1*C_conv_l1,C_conv_l2,'R',2*sl_dec);
[p_c_1_N2,Eb] = bp_clb(p_c_1_N2,Eb,hm_c_1_N2,ym_flat,K_conv_l1,C_conv_l1,nl(7),K_conv_l1*C_conv_l1,'R',2*sl_dec);

% residual modification
Eb = Eb + Ebp;
Eb1 = [Eb(1:2:end,:) Eb(2:2:end,:)];

% Network 1

[E] = get_oplayer_error(Y,ym_1,sl_dec,cfn);
E = E + Eb1;

E1 = E(:,1:dout);
E2 = E(:,dout+1:2*dout);

% Backprop top
p_f_4_1_dec.gU = (E1'*hfmd3);
p_f_4_1_dec.gbu = sum(E1,1)';

p_f_4_2_dec.gU = (E2'*hfmd3);
p_f_4_2_dec.gbu = sum(E2,1)';

Ebf = E1*p_f_4_1_dec.U + E2*p_f_4_2_dec.U;
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_3_dec,zfmd3,ifmd3,ffmd3,cfmd3,ofmd3,hcfmd3,nl(6),sl_dec,'frnn');
[p_lf_3_dec] = gradients_lstm(dec_rnn_ip_2,p_lf_3_dec,hfmd3,cfmd3,dom,dfm,dim,dzm,nl(6),'frnn');

Ebfl = Ebf(:,1:nl(6));
Ebfa_2 = Ebf(:,nl(6)+1:nl(6)+2*nl(2));

[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebfl',p_lf_2_dec,zfmd2,ifmd2,ffmd2,cfmd2,ofmd2,hcfmd2,nl(5),sl_dec,'frnn');
[p_lf_2_dec] = gradients_lstm(dec_rnn_ip_1,p_lf_2_dec,hfmd2,cfmd2,dom,dfm,dim,dzm,nl(5),'frnn');

Ebfl = Ebf(:,1:nl(5));
Ebfa_1 = Ebf(:,nl(5)+1:nl(5)+2*nl(2));

Eb = Ebfa_1 + Ebfa_2;
[p_f_3_0,p_f_3_1,p_f_3_2,Eb1,Eb2] = bp_att(Eb,sm,H,hm_f_3_1',hfmd1',p_f_3_1,p_f_3_2,p_f_3_0,nl(4),nl(2),nl(3),sl_dec,sl_enc);

Eb = Ebfl + Eb1;
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Eb',p_lf_1_dec,zfmd1,ifmd1,ffmd1,cfmd1,ofmd1,hcfmd1,nl(4),sl_dec,'frnn');
[p_lf_1_dec] = gradients_lstm(Yp,p_lf_1_dec,hfmd1,cfmd1,dom,dfm,dim,dzm,nl(4),'frnn');


% Encoder
if enc_layers == 3

Ebf = Eb2(:,1:nl(2));
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_3,zfm3,ifm3,ffm3,cfm3,ofm3,hcfm3,nl(2),sl_enc,'frnn');
[p_lf_3] = gradients_lstm(hfm2,p_lf_3,hfm3,cfm3,dom,dfm,dim,dzm,nl(2),'frnn');

Ebb = Eb2(:,nl(2)+1:2*nl(2));
[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb',p_lb_3,zbm3,ibm3,fbm3,cbm3,obm3,hcbm3,nl(2),sl_enc,'brnn');
[p_lb_3] = gradients_lstm(hbm2,p_lb_3,hbm3,cbm3,dom,dfm,dim,dzm,nl(2),'brnn');


[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_2,zfm2,ifm2,ffm2,cfm2,ofm2,hcfm2,nl(2),sl_enc,'frnn');
[p_lf_2] = gradients_lstm(hfm1,p_lf_2,hfm2,cfm2,dom,dfm,dim,dzm,nl(2),'frnn');

[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb',p_lb_2,zbm2,ibm2,fbm2,cbm2,obm2,hcbm2,nl(2),sl_enc,'brnn');
[p_lb_2] = gradients_lstm(hbm1,p_lb_2,hbm2,cbm2,dom,dfm,dim,dzm,nl(2),'brnn');


[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_1,zfm1,ifm1,ffm1,cfm1,ofm1,hcfm1,nl(2),sl_enc,'frnn');
[p_lf_1] = gradients_lstm(X,p_lf_1,hfm1,cfm1,dom,dfm,dim,dzm,nl(2),'frnn');

[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb',p_lb_1,zbm1,ibm1,fbm1,cbm1,obm1,hcbm1,nl(2),sl_enc,'brnn');
[p_lb_1] = gradients_lstm(X,p_lb_1,hbm1,cbm1,dom,dfm,dim,dzm,nl(2),'brnn');


Eb = Ebf + Ebb;

elseif enc_layers == 2

Ebf = Eb2(:,1:nl(2));

[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_2,zfm2,ifm2,ffm2,cfm2,ofm2,hcfm2,nl(2),sl_enc,'frnn');
[p_lf_2] = gradients_lstm(hfm1,p_lf_2,hfm2,cfm2,dom,dfm,dim,dzm,nl(2),'frnn');

Ebb = Eb2(:,nl(2)+1:2*nl(2));

[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb',p_lb_2,zbm2,ibm2,fbm2,cbm2,obm2,hcbm2,nl(2),sl_enc,'brnn');
[p_lb_2] = gradients_lstm(hbm1,p_lb_2,hbm2,cbm2,dom,dfm,dim,dzm,nl(2),'brnn');


[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_1,zfm1,ifm1,ffm1,cfm1,ofm1,hcfm1,nl(2),sl_enc,'frnn');
[p_lf_1] = gradients_lstm(X,p_lf_1,hfm1,cfm1,dom,dfm,dim,dzm,nl(2),'frnn');

[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb',p_lb_1,zbm1,ibm1,fbm1,cbm1,obm1,hcbm1,nl(2),sl_enc,'brnn');
[p_lb_1] = gradients_lstm(X,p_lb_1,hbm1,cbm1,dom,dfm,dim,dzm,nl(2),'brnn');

Eb = Ebf + Ebb;


elseif enc_layers == 1

Ebf = Eb2(:,1:nl(2));

[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_1,zfm1,ifm1,ffm1,cfm1,ofm1,hcfm1,nl(2),sl_enc,'frnn');
[p_lf_1] = gradients_lstm(X,p_lf_1,hfm1,cfm1,dom,dfm,dim,dzm,nl(2),'frnn');

Ebb = Eb2(:,nl(2)+1:2*nl(2));

[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb',p_lb_1,zbm1,ibm1,fbm1,cbm1,obm1,hcbm1,nl(2),sl_enc,'brnn');
[p_lb_1] = gradients_lstm(X,p_lb_1,hbm1,cbm1,dom,dfm,dim,dzm,nl(2),'brnn');

Eb = Ebf + Ebb;

end





