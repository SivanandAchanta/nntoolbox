
[E] = get_oplayer_error(Y,ym,sl_dec,cfn);

E1 = E(:,1:dout);
E2 = E(:,dout+1:2*dout);

fprintf('Error  %f %f ',max(sum(E1.^2,2)),max(sum(E2.^2,2)));

% Backprop top
p_f_4_1_dec.gU = (E1'*hfmd3);
p_f_4_1_dec.gbu = sum(E1,1)';

p_f_4_2_dec.gU = (E2'*hfmd3);
p_f_4_2_dec.gbu = sum(E2,1)';

Ebf = E1*p_f_4_1_dec.U + E2*p_f_4_2_dec.U;
fprintf('%f ',max(sum(Ebf.^2,2)));

[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf',p_lf_3_dec,zfmd3,ifmd3,ffmd3,cfmd3,ofmd3,hcfmd3,nl(8),sl_dec,'frnn');
[p_lf_3_dec] = gradients_lstm(dec_rnn_ip_2,p_lf_3_dec,hfmd3,cfmd3,dom,dfm,dim,dzm,nl(8),'frnn');

Ebfl = Ebf(:,1:nl(8));
Ebfa_2 = Ebf(:,nl(8)+1:nl(8)+2*nl(2));
fprintf('%f ',max(sum(Ebfl.^2,2)));

[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebfl',p_lf_2_dec,zfmd2,ifmd2,ffmd2,cfmd2,ofmd2,hcfmd2,nl(7),sl_dec,'frnn');
[p_lf_2_dec] = gradients_lstm(dec_rnn_ip_1,p_lf_2_dec,hfmd2,cfmd2,dom,dfm,dim,dzm,nl(7),'frnn');

Ebfl = Ebf(:,1:nl(7));
Ebfa_1 = Ebf(:,nl(7)+1:nl(7)+2*nl(2));
Eb = Ebfa_1 + Ebfa_2;
fprintf('%f %f ',max(sum(Ebfa_1.^2,2)),max(sum(Ebfa_2.^2,2)));

Eb = Eb*sl_enc;
[p_f_3_0,p_f_3_1,p_f_3_2,Eb1,Eb2] = bp_att(Eb,sm,H,hm_f_3_1',hfmd1',p_f_3_1,p_f_3_2,p_f_3_0,nl(6),nl(2),nl(3),sl_dec,sl_enc);

%Eb1 = Eb1*sl_enc;
Eb = Ebfl + Eb1;
fprintf('%f %f ',max(sum(Ebfl.^2,2)),max(sum(Eb1.^2,2)));
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Eb',p_lf_1_dec,zfmd1,ifmd1,ffmd1,cfmd1,ofmd1,hcfmd1,nl(6),sl_dec,'frnn');
[p_lf_1_dec] = gradients_lstm(hm_f_2,p_lf_1_dec,hfmd1,cfmd1,dom,dfm,dim,dzm,nl(6),'frnn');

Eb = bp_dropout(Ebf,dm_f_2);
[p_f_2.gU,p_f_2.gbu,Eb] = bp_cpu_ll(nl(5),f(4),Eb,hm_f_2,hm_f_1,p_f_2.U,sl_dec);
Eb = bp_dropout(Eb,dm_f_1);
[p_f_1.gU,p_f_1.gbu,Eb] = bp_cpu_ll(nl(4),f(3),Eb,hm_f_1,Yp,p_f_1.U,sl_dec);

%Eb2 = Eb2*sl_dec;
fprintf('%f \n',max(sum(Eb2.^2,2)));

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





