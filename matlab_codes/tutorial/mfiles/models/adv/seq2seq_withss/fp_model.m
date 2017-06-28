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

hm_f_3_1 = fp_cpu_ll_ow(H,p_f_3_1,'L');

ss_vec = binornd(1,ss_prob,[1 sl_dec]);

hp_dec1 = zeros(nl(4),1);
hp_dec2 = zeros(nl(5),1);
hp_dec3 = zeros(nl(6),1);

cp_dec1 = zeros(nl(4),1);
cp_dec2 = zeros(nl(5),1);
cp_dec3 = zeros(nl(6),1);

zfmd1 = zeros(sl_dec,nl(4));
ifmd1 = zeros(sl_dec,nl(4));
ffmd1 = zeros(sl_dec,nl(4));
cfmd1 = zeros(sl_dec,nl(4));
ofmd1 = zeros(sl_dec,nl(4));
hcfmd1 = zeros(sl_dec,nl(4));
hfmd1 = zeros(sl_dec,nl(4));

zfmd2 = zeros(sl_dec,nl(5));
ifmd2 = zeros(sl_dec,nl(5));
ffmd2 = zeros(sl_dec,nl(5));
cfmd2 = zeros(sl_dec,nl(5));
ofmd2 = zeros(sl_dec,nl(5));
hcfmd2 = zeros(sl_dec,nl(5));
hfmd2 = zeros(sl_dec,nl(5));

zfmd3 = zeros(sl_dec,nl(6));
ifmd3 = zeros(sl_dec,nl(6));
ffmd3 = zeros(sl_dec,nl(6));
cfmd3 = zeros(sl_dec,nl(6));
ofmd3 = zeros(sl_dec,nl(6));
hcfmd3 = zeros(sl_dec,nl(6));
hfmd3 = zeros(sl_dec,nl(6));


bm = (zeros(sl_dec,size(hm_f_3_1,1)));
sm = (zeros(sl_dec,size(hm_f_3_1,1)));
cm = (zeros(sl_dec,nl(3)));

dec_rnn_ip_1 = zeros(sl_dec,nl(4) + 2*nl(2));
dec_rnn_ip_2 = zeros(sl_dec,nl(5) + 2*nl(2));

ym = zeros(sl_dec,2*dout);
Ypi = zeros(size(Yp));

for dts = 1:sl_dec

if ss_vec(dts) == 1 || dts == 1
   yp = Yp(dts,:);
else
   yp = ym2;
end

% Decoder
[zfmd1(dts,:),ifmd1(dts,:),ffmd1(dts,:),cfmd1(dts,:),ofmd1(dts,:),hcfmd1(dts,:),hfmd1(dts,:)] = fp_lstm_singlestep(yp,hp_dec1,cp_dec1,p_lf_1_dec,nl(4),'frnn');

% Attention Layer
[bm(dts,:),sm(dts,:),cm(dts,:)] = fp_att_singlestep(H',hm_f_3_1',hfmd1(dts,:)',p_f_3_2,p_f_3_0,nl(3));

dec_rnn_ip_1(dts,:) = [hfmd1(dts,:) cm(dts,:)];
[zfmd2(dts,:),ifmd2(dts,:),ffmd2(dts,:),cfmd2(dts,:),ofmd2(dts,:),hcfmd2(dts,:),hfmd2(dts,:)] = fp_lstm_singlestep(dec_rnn_ip_1(dts,:),hp_dec2,cp_dec2,p_lf_2_dec,nl(5),'frnn');
dec_rnn_ip_2(dts,:) = [hfmd2(dts,:) cm(dts,:)];
[zfmd3(dts,:),ifmd3(dts,:),ffmd3(dts,:),cfmd3(dts,:),ofmd3(dts,:),hcfmd3(dts,:),hfmd3(dts,:)] = fp_lstm_singlestep(dec_rnn_ip_2(dts,:),hp_dec3,cp_dec3,p_lf_3_dec,nl(6),'frnn');

hp_dec1 = hfmd1(dts,:)';
cp_dec1 = cfmd1(dts,:)';
hp_dec2 = hfmd2(dts,:)';
cp_dec2 = cfmd2(dts,:)';
hp_dec3 = hfmd3(dts,:)';
cp_dec3 = cfmd3(dts,:)';

% Final Output Layer
ac = p_f_4_1_dec.U*hfmd3(dts,:)';        
ym1 = bsxfun(@plus,ac,p_f_4_1_dec.bu)';
ac = p_f_4_2_dec.U*hfmd3(dts,:)';        
ym2 = bsxfun(@plus,ac,p_f_4_2_dec.bu)';

ym(dts,:) = [ym1 ym2];
Ypi(dts,:) = yp';

end


