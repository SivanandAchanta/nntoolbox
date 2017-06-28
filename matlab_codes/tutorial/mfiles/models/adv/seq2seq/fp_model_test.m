% Encoder

% 1/2/3 - BLSTM Layers
if enc_layers == 1
   [zfm1,ifm1,ffm1,cfm1,ofm1,hcfm1,hfm1] = fp_lstm(X,p_lf_1,nl(2),sl_enc,'frnn');
   [zbm1,ibm1,fbm1,cbm1,obm1,hcbm1,hbm1] = fp_lstm(X,p_lb_1,nl(2),sl_enc,'brnn');
   H = [hfm1 hbm1];

elseif enc_layers == 2
   [zfm1,ifm1,ffm1,cfm1,ofm1,hcfm1,hfm1] = fp_lstm(X,p_lf_1,nl(2),sl_enc,'frnn');
   [zbm1,ibm1,fbm1,cbm1,obm1,hcbm1,hbm1] = fp_lstm(X,p_lb_1,nl(2),sl_enc,'brnn');
   [zfm2,ifm2,ffm2,cfm2,ofm2,hcfm2,hfm2] = fp_lstm(hfm1,p_lf_2,nl(2),sl_enc,'frnn');
   [zbm2,ibm2,fbm2,cbm2,obm2,hcbm2,hbm2] = fp_lstm(hbm1,p_lb_2,nl(2),sl_enc,'brnn');
   H = [hfm2 hbm2];

elseif enc_layers == 3
   [zfm1,ifm1,ffm1,cfm1,ofm1,hcfm1,hfm1] = fp_lstm(X,p_lf_1,nl(2),sl_enc,'frnn');
   [zbm1,ibm1,fbm1,cbm1,obm1,hcbm1,hbm1] = fp_lstm(X,p_lb_1,nl(2),sl_enc,'brnn');
   [zfm2,ifm2,ffm2,cfm2,ofm2,hcfm2,hfm2] = fp_lstm(hfm1,p_lf_2,nl(2),sl_enc,'frnn');
   [zbm2,ibm2,fbm2,cbm2,obm2,hcbm2,hbm2] = fp_lstm(hbm1,p_lb_2,nl(2),sl_enc,'brnn');
   [zfm3,ifm3,ffm3,cfm3,ofm3,hcfm3,hfm3] = fp_lstm(hfm2,p_lf_3,nl(2),sl_enc,'frnn');
   [zbm3,ibm3,fbm3,cbm3,obm3,hcbm3,hbm3] = fp_lstm(hbm2,p_lb_3,nl(2),sl_enc,'brnn');
   H = [hfm3 hbm3];

end 


% Decoder

[zfmd1,ifmd1,ffmd1,cfmd1,ofmd1,hcfmd1,hfmd1] = fp_lstm(Yp,p_lf_1_dec,nl(4),sl_dec,'frnn');

% Attention Layer
hm_f_3_1 = fp_cpu_ll_ow(H,p_f_3_1,'L');
[bm,sm,cm] = fp_att(H',hm_f_3_1',hfmd1',p_f_3_2,p_f_3_0,nl(4),sl_dec);

dec_rnn_ip_1 = [hfmd1 cm];
[zfmd2,ifmd2,ffmd2,cfmd2,ofmd2,hcfmd2,hfmd2] = fp_lstm(dec_rnn_ip_1,p_lf_2_dec,nl(5),sl_dec,'frnn');
dec_rnn_ip_2 = [hfmd2 cm];
[zfmd3,ifmd3,ffmd3,cfmd3,ofmd3,hcfmd3,hfmd3] = fp_lstm(dec_rnn_ip_2,p_lf_3_dec,nl(6),sl_dec,'frnn');

% Final Output Layer
ac = p_f_4_1_dec.U*hfmd3';        
ym1 = bsxfun(@plus,ac,p_f_4_1_dec.bu)';
ac = p_f_4_2_dec.U*hfmd3';        
ym2 = bsxfun(@plus,ac,p_f_4_2_dec.bu)';

ym = [ym1 ym2];

