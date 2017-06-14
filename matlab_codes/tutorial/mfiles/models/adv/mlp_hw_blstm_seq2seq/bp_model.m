
% Error at Output Layer
switch cfn
    case 'ls'
        E   = -(Y - ym)/sl_dec;
    case  'nll'
        E   = -(Y - ym)/sl_dec;
end
E1 = E(:,1:dout);
E2 = E(:,dout+1:2*dout);

% Backprop top
p_f_4_1_dec.gU = (E1'*hfm3);
p_f_4_1_dec.gbu = sum(E1,1)';

p_f_4_2_dec.gU = (E2'*hfm3);
p_f_4_2_dec.gbu = sum(E2,1)';

Ebf = E1*p_f_4_1_dec.U + E2*p_f_4_2_dec.U;
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_3_dec,zfm3,ifm3,ffm3,cfm3,ofm3,hcfm3,nl(14),sl_dec,'frnn');
[p_lf_3_dec] = gradients_lstm(dec_rnn_ip_2,p_lf_3_dec,hfm3,cfm3,dom,dfm,dim,dzm,nl(14),'frnn');

Ebfl = Ebf(:,1:nl(13));
Ebfa_2 = Ebf(:,nl(13)+1:nl(13)+nl(9));

[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebfl',p_lf_2_dec,zfm2,ifm2,ffm2,cfm2,ofm2,hcfm2,nl(13),sl_dec,'frnn');
[p_lf_2_dec] = gradients_lstm(dec_rnn_ip_1,p_lf_2_dec,hfm2,cfm2,dom,dfm,dim,dzm,nl(13),'frnn');

Ebfl = Ebf(:,1:nl(12));
Ebfa_1 = Ebf(:,nl(12)+1:nl(12)+nl(9));

Eb = Ebfa_1 + Ebfa_2;
[p_f_3_0,p_f_3_1,p_f_3_2,Eb1,Eb2] = bp_att(Eb,sm,H,hm_f_3_1',hfm1',p_f_3_1,p_f_3_2,p_f_3_0,nl(10),nl(8),nl(9),sl_dec,sl_enc);

Eb = Ebfl + Eb1;
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Eb',p_lf_1_dec,zfm1,ifm1,ffm1,cfm1,ofm1,hcfm1,nl(12),sl_dec,'frnn');
[p_lf_1_dec] = gradients_lstm(hm_f_2_dec,p_lf_1_dec,hfm1,cfm1,dom,dfm,dim,dzm,nl(12),'frnn');

Eb = bp_dropout(Ebf,dm_f_2_dec);
[p_f_2_dec.gU,p_f_2_dec.gbu,Eb] = bp_cpu_ll(nl(11),f(10),Eb,hm_f_2_dec,hm_f_1_dec,p_f_2_dec.U,sl_dec);
Eb = bp_dropout(Eb,dm_f_1_dec);
[p_f_1_dec.gU,p_f_1_dec.gbu,Eb] = bp_cpu_ll(nl(10),f(9),Eb,hm_f_1_dec,Yn,p_f_1_dec.U,sl_dec);

Eb2

% Encoder
Ebf = Eb2(:,1:nl(8));
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_1,zfm,ifm,ffm,cfm,ofm,hcfm,nl(8),sl_enc,'frnn');
[p_lf_1] = gradients_lstm(hm_h_4,p_lf_1,hfm,cfm,dom,dfm,dim,dzm,nl(8),'frnn');

Ebb = Eb2(:,nl(8)+1:2*nl(8));
[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb',p_lb_1,zbm,ibm,fbm,cbm,obm,hcbm,nl(8),sl_enc,'brnn');
[p_lb_1] = gradients_lstm(hm_h_4,p_lb_1,hbm,cbm,dom,dfm,dim,dzm,nl(8),'brnn');

Eb = Ebf + Ebb;

[p_h_4.gW,p_h_4.gb,p_h_4.gWt,p_h_4.gbt,Eb] = bp_hl(nl(7),sl_enc,Eb,p_h_4,htm_h_4,tm_h_4,hm_h_3,f(3));
[p_h_3.gW,p_h_3.gb,p_h_3.gWt,p_h_3.gbt,Eb] = bp_hl(nl(6),sl_enc,Eb,p_h_3,htm_h_3,tm_h_3,hm_h_2,f(4));
[p_h_2.gW,p_h_2.gb,p_h_2.gWt,p_h_2.gbt,Eb] = bp_hl(nl(5),sl_enc,Eb,p_h_2,htm_h_2,tm_h_2,hm_h_1,f(5));
[p_h_1.gW,p_h_1.gb,p_h_1.gWt,p_h_1.gbt,Eb] = bp_hl(nl(4),sl_enc,Eb,p_h_1,htm_h_1,tm_h_1,hm_f_2,f(6));

Eb = bp_dropout(Eb,dm_f_2);
[p_f_2.gU,p_f_2.gbu,Eb] = bp_cpu_ll(nl(3),f(2),Eb,hm_f_2,hm_f_1,p_f_2.U,sl_enc);
Eb = bp_dropout(Eb,dm_f_1);
[p_f_1.gU,p_f_1.gbu,Eb] = bp_cpu_ll(nl(2),f(1),Eb,hm_f_1,X,p_f_1.U,sl_enc);

