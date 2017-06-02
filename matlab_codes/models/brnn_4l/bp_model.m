% bp
switch cfn
    case 'ls'
        E   = -(Y - ym)/sl;
    case  'nll'
        E   = -(Y - ym)/sl;
end

% Backprop top layer
Gpo_2.gU = (E'*hcm_rb_4);
Gpo_2.gbu = sum(E,1)';
Ebb = E*Gpo_2.U;

Gpo_1.gU = (E'*hcm_rf_4);
Ebf = E*Gpo_1.U;

[Gp_rb_4.gWi,Gp_rb_4.gWfr,Gp_rb_4.gbh,Ebb] = bp_cpu_rl(nl(6),f(5),Ebb',hcm_rb_4,hcm_rb_3,Gp_rb_4,sl,'brnn');
[Gp_rf_4.gWi,Gp_rf_4.gWfr,Gp_rf_4.gbh,Ebf] = bp_cpu_rl(nl(6),f(5),Ebf',hcm_rf_4,hcm_rf_3,Gp_rf_4,sl,'frnn');
[Gp_rb_3.gWi,Gp_rb_3.gWfr,Gp_rb_3.gbh,Ebb] = bp_cpu_rl(nl(5),f(4),Ebb',hcm_rb_3,hcm_rb_2,Gp_rb_3,sl,'brnn');
[Gp_rf_3.gWi,Gp_rf_3.gWfr,Gp_rf_3.gbh,Ebf] = bp_cpu_rl(nl(5),f(4),Ebf',hcm_rf_3,hcm_rf_2,Gp_rf_3,sl,'frnn');
[Gp_rb_2.gWi,Gp_rb_2.gWfr,Gp_rb_2.gbh,Ebb] = bp_cpu_rl(nl(4),f(3),Ebb',hcm_rb_2,hcm_rb_1,Gp_rb_2,sl,'brnn');
[Gp_rf_2.gWi,Gp_rf_2.gWfr,Gp_rf_2.gbh,Ebf] = bp_cpu_rl(nl(4),f(3),Ebf',hcm_rf_2,hcm_rf_1,Gp_rf_2,sl,'frnn');
[Gp_rb_1.gWi,Gp_rb_1.gWfr,Gp_rb_1.gbh,Ebb] = bp_cpu_rl(nl(3),f(2),Ebb',hcm_rb_1,hcm_i_1,Gp_rb_1,sl,'brnn');
[Gp_rf_1.gWi,Gp_rf_1.gWfr,Gp_rf_1.gbh,Ebf] = bp_cpu_rl(nl(3),f(2),Ebf',hcm_rf_1,hcm_i_1,Gp_rf_1,sl,'frnn');

Eb = Ebf + Ebb;
[Gpi_1.gU,Gpi_1.gbu,Eb] = bp_cpu_ll(nl(2),f(1),Eb,hcm_i_1,X,Gpi_1.U,sl);

