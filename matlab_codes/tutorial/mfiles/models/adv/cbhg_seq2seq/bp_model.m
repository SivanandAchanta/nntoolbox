% Network 1

[E] = get_oplayer_error(Y,ym,sl_dec,cfn);

E1 = E(:,1:dout);
E2 = E(:,dout+1:2*dout);

% Backprop top
p_f_4_1_dec.gU = (E1'*hfmd3);
p_f_4_1_dec.gbu = sum(E1,1)';

p_f_4_2_dec.gU = (E2'*hfmd3);
p_f_4_2_dec.gbu = sum(E2,1)';

Ebf = E1*p_f_4_1_dec.U + E2*p_f_4_2_dec.U;
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_3_dec,zfmd3,ifmd3,ffmd3,cfmd3,ofmd3,hcfmd3,nl(11),sl_dec,'frnn');
[p_lf_3_dec] = gradients_lstm(dec_rnn_ip_2,p_lf_3_dec,hfmd3,cfmd3,dom,dfm,dim,dzm,nl(11),'frnn');

Ebfl = Ebf(:,1:nl(11));
Ebfa_2 = Ebf(:,nl(11)+1:nl(11)+2*nl(5));

[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebfl',p_lf_2_dec,zfmd2,ifmd2,ffmd2,cfmd2,ofmd2,hcfmd2,nl(10),sl_dec,'frnn');
[p_lf_2_dec] = gradients_lstm(dec_rnn_ip_1,p_lf_2_dec,hfmd2,cfmd2,dom,dfm,dim,dzm,nl(10),'frnn');

Ebfl = Ebf(:,1:nl(10));
Ebfa_1 = Ebf(:,nl(10)+1:nl(10)+2*nl(5));

Eb = Ebfa_1 + Ebfa_2;
[p_f_3_0,p_f_3_1,p_f_3_2,Eb1,Eb2] = bp_att(Eb,sm,H,hm_f_3_1',hfmd1',p_f_3_1,p_f_3_2,p_f_3_0,nl(9),nl(5),nl(6),sl_dec,sl_enc);

Eb = Ebfl + Eb1;
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Eb',p_lf_1_dec,zfmd1,ifmd1,ffmd1,cfmd1,ofmd1,hcfmd1,nl(9),sl_dec,'frnn');
[p_lf_1_dec] = gradients_lstm(hm_f_2,p_lf_1_dec,hfmd1,cfmd1,dom,dfm,dim,dzm,nl(9),'frnn');


Eb = bp_dropout(Ebf,dm_f_2);
[p_f_2.gU,p_f_2.gbu,Eb] = bp_cpu_ll(nl(8),f(7),Eb,hm_f_2,hm_f_1,p_f_2.U,sl_dec);
Eb = bp_dropout(Eb,dm_f_1);
[p_f_1.gU,p_f_1.gbu,Eb] = bp_cpu_ll(nl(7),f(6),Eb,hm_f_1,Yp,p_f_1.U,sl_dec);


% Encoder
Ebf = Eb2(:,1:nl(5));
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_1_N1,zfmn1,ifmn1,ffmn1,cfmn1,ofmn1,hcfmn1,nl(5),sl_enc,'frnn');
[p_lf_1_N1] = gradients_lstm(hm_h_4_N1,p_lf_1_N1,hfmn1,cfmn1,dom,dfm,dim,dzm,nl(5),'frnn');

Ebb = Eb2(:,nl(5)+1:2*nl(5));
[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb',p_lb_1_N1,zbmn1,ibmn1,fbmn1,cbmn1,obmn1,hcbmn1,nl(5),sl_enc,'brnn');
[p_lb_1_N1] = gradients_lstm(hm_h_4_N1,p_lb_1_N1,hbmn1,cbmn1,dom,dfm,dim,dzm,nl(5),'brnn');

Eb = Ebf + Ebb;
% Eb = Ebf;

[p_h_4_N1.gW,p_h_4_N1.gb,p_h_4_N1.gWt,p_h_4_N1.gbt,Eb] = bp_hl(nl(4),sl_enc,Eb,p_h_4_N1,htm_h_4_N1,tm_h_4_N1,hm_h_3_N1,f(3));
[p_h_3_N1.gW,p_h_3_N1.gb,p_h_3_N1.gWt,p_h_3_N1.gbt,Eb] = bp_hl(nl(4),sl_enc,Eb,p_h_3_N1,htm_h_3_N1,tm_h_3_N1,hm_h_2_N1,f(3));
[p_h_2_N1.gW,p_h_2_N1.gb,p_h_2_N1.gWt,p_h_2_N1.gbt,Eb] = bp_hl(nl(4),sl_enc,Eb,p_h_2_N1,htm_h_2_N1,tm_h_2_N1,hm_h_1_N1,f(3));
[p_h_1_N1.gW,p_h_1_N1.gb,p_h_1_N1.gWt,p_h_1_N1.gbt,Eb] = bp_hl(nl(4),sl_enc,Eb,p_h_1_N1,htm_h_1_N1,tm_h_1_N1,hm_c_4_N1,f(3));

% convulution layers with residual connection
Ebp = Eb;
[p_c_3_N1,Eb] = bp_cl(p_c_3_N1,Eb,hm_c_3_N1,hm_c_2_N1,K_conv_l2_n1,C_conv_l2_n1,C_conv_l2_n1,'L',sl_enc);
[p_c_2_N1,Eb] = bp_cl(p_c_2_N1,Eb,hm_c_2_N1,hm_c_1_N1,K_conv_l2_n1,K_conv_l1_n1*C_conv_l1_n1,C_conv_l2_n1,'R',sl_enc);
[p_c_1_N1,Eb] = bp_clb(p_c_1_N1,Eb,hm_c_1_N1,hm_f_2_N1,K_conv_l1_n1,C_conv_l1_n1,nl(3),K_conv_l1_n1*C_conv_l1_n1,'R',sl_enc);

% pre-net
Eb = Eb + Ebp; % residual net modification
Eb = bp_dropout(Eb,dm_f_2_N1);
[p_f_2_N1.gU,p_f_2_N1.gbu,Eb] = bp_cpu_ll(nl(3),f(2),Eb,hm_f_2_N1,hm_f_1_N1,p_f_2_N1.U,sl_enc);
Eb = bp_dropout(Eb,dm_f_1_N1);
[p_f_1_N1.gU,p_f_1_N1.gbu,Eb] = bp_cpu_ll(nl(2),f(1),Eb,hm_f_1_N1,X,p_f_1_N1.U,sl_enc);




