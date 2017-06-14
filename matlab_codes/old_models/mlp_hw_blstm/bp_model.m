
% Error at Output Layer
switch cfn
    case 'ls'
        E   = -(Y - ym)/sl;
    case  'nll'
        E   = -(Y - ym)/sl;
end

% Backprop top ms
p_f_3_1.gU = (E'*hfm);
p_f_3_2.gU = (E'*hbm);
p_f_3_1.gbu = sum(E,1)';

Ebf = p_f_3_1.U'*E';
[dhm,dom,dcm,dfm,dim,dzm,Ebf] = bp_lstm(Ebf,p_lf_1,zfm,ifm,ffm,cfm,ofm,hcfm,nl(8),sl,'frnn');
[p_lf_1] = gradients_lstm(hm_h_4,p_lf_1,hfm,cfm,dom,dfm,dim,dzm,nl(8),'frnn');

Ebb = p_f_3_2.U'*E';
[dhm,dom,dcm,dfm,dim,dzm,Ebb] = bp_lstm(Ebb,p_lb_1,zbm,ibm,fbm,cbm,obm,hcbm,nl(8),sl,'brnn');
[p_lb_1] = gradients_lstm(hm_h_4,p_lb_1,hbm,cbm,dom,dfm,dim,dzm,nl(8),'brnn');

Eb = Ebf + Ebb;
% Eb = Ebf;

[p_h_4.gW,p_h_4.gb,p_h_4.gWt,p_h_4.gbt,Eb] = bp_hl(nl(7),sl,Eb,p_h_4,htm_h_4,tm_h_4,hm_h_3,f(3));
[p_h_3.gW,p_h_3.gb,p_h_3.gWt,p_h_3.gbt,Eb] = bp_hl(nl(6),sl,Eb,p_h_3,htm_h_3,tm_h_3,hm_h_2,f(4));
[p_h_2.gW,p_h_2.gb,p_h_2.gWt,p_h_2.gbt,Eb] = bp_hl(nl(5),sl,Eb,p_h_2,htm_h_2,tm_h_2,hm_h_1,f(5));
[p_h_1.gW,p_h_1.gb,p_h_1.gWt,p_h_1.gbt,Eb] = bp_hl(nl(4),sl,Eb,p_h_1,htm_h_1,tm_h_1,hm_f_2,f(6));

Eb = bp_dropout(Eb,dm_f_2);
[p_f_2.gU,p_f_2.gbu,Eb] = bp_cpu_ll(nl(3),f(2),Eb,hm_f_2,hm_f_1,p_f_2.U,sl);
Eb = bp_dropout(Eb,dm_f_1);
[p_f_1.gU,p_f_1.gbu,Eb] = bp_cpu_ll(nl(2),f(1),Eb,hm_f_1,X,p_f_1.U,sl);

