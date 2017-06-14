
% 1 - BLSTM Layer
[zfm1,ifm1,ffm1,cfm1,ofm1,hcfm1,hfm1] = fp_lstm(X,p_lf_1,nl(2),sl,'frnn');
[zbm1,ibm1,fbm1,cbm1,obm1,hcbm1,hbm1] = fp_lstm(X,p_lb_1,nl(2),sl,'brnn');

% O/P Layer
ym1 = fp_cpu_ll_ow(hfm1,p_f_1,'L');
ym2 = fp_cpu_ll(hbm1,p_f_2,'L');

ym = ym1 + ym2;
ym = get_actf(f(end),ym);

