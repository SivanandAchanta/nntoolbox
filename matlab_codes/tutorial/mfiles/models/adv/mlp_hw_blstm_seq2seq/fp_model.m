% Encoder
% 2 - feedforward layers with dropout (pre-net)
hm_f_1 = fp_cpu_ll(X,p_f_1,f(1));
[hm_f_1,dm_f_1] = fp_dropout(hm_f_1,dp(1),nl(2));
hm_f_2 = fp_cpu_ll(hm_f_1,p_f_2,f(2));
[hm_f_2,dm_f_2] = fp_dropout(hm_f_2,dp(2),nl(3));

% 4 - Highway layers
[tm_h_1,htm_h_1,hm_h_1] = fp_hl(hm_f_2,p_h_1,f(3));
[tm_h_2,htm_h_2,hm_h_2] = fp_hl(hm_h_1,p_h_2,f(4));
[tm_h_3,htm_h_3,hm_h_3] = fp_hl(hm_h_2,p_h_3,f(5));
[tm_h_4,htm_h_4,hm_h_4] = fp_hl(hm_h_3,p_h_4,f(6));

% 1 - BLSTM Layer
[zfm,ifm,ffm,cfm,ofm,hcfm,hfm] = fp_lstm(hm_h_4,p_lf_1,nl(8),sl_enc,'frnn');
[zbm,ibm,fbm,cbm,obm,hcbm,hbm] = fp_lstm(hm_h_4,p_lb_1,nl(8),sl_enc,'brnn');

% Decoder
% 2 - feedforward layers with dropout (pre-net)
hm_f_1_dec = fp_cpu_ll(Yn,p_f_1_dec,f(9));
[hm_f_1_dec,dm_f_1_dec] = fp_dropout(hm_f_1_dec,dp(3),nl(10));
hm_f_2_dec = fp_cpu_ll(hm_f_1_dec,p_f_2_dec,f(10));
[hm_f_2_dec,dm_f_2_dec] = fp_dropout(hm_f_2_dec,dp(4),nl(11));

[zfm1,ifm1,ffm1,cfm1,ofm1,hcfm1,hfm1] = fp_lstm(hm_f_2_dec,p_lf_1_dec,nl(12),sl_dec,'frnn');

% Attention Layer
H = [hfm hbm];
hm_f_3_1 = fp_cpu_ll_ow(H,p_f_3_1,'L');
[bm,sm,cm] = fp_att(H',hm_f_3_1',hfm1',p_f_3_2,p_f_3_0,nl(9),sl_dec);

dec_rnn_ip_1 = [hfm1 cm];
[zfm2,ifm2,ffm2,cfm2,ofm2,hcfm2,hfm2] = fp_lstm(dec_rnn_ip_1,p_lf_2_dec,nl(13),sl_dec,'frnn');
dec_rnn_ip_2 = [hfm2 cm];
[zfm3,ifm3,ffm3,cfm3,ofm3,hcfm3,hfm3] = fp_lstm(dec_rnn_ip_2,p_lf_3_dec,nl(14),sl_dec,'frnn');

% Final Output Layer
ac = p_f_4_1_dec.U*hfm3';        
ym1 = bsxfun(@plus,ac,p_f_4_1_dec.bu)';
ac = p_f_4_2_dec.U*hfm3';        
ym2 = bsxfun(@plus,ac,p_f_4_2_dec.bu)';

ym = [ym1 ym2];

