hcm_i_1 = fp_cpu_ll(X,Gpi_1,f(1));

hcm_rf_1 = fp_cpu_rl(hcm_i_1,Gp_rf_1,f(2),nl(3),sl,'frnn');
hcm_rb_1 = fp_cpu_rl(hcm_i_1,Gp_rb_1,f(2),nl(3),sl,'brnn');
hcm_rf_2 = fp_cpu_rl(hcm_rf_1,Gp_rf_2,f(3),nl(4),sl,'frnn');
hcm_rb_2 = fp_cpu_rl(hcm_rb_1,Gp_rb_2,f(3),nl(4),sl,'brnn');
hcm_rf_3 = fp_cpu_rl(hcm_rf_2,Gp_rf_3,f(4),nl(5),sl,'frnn');
hcm_rb_3 = fp_cpu_rl(hcm_rb_2,Gp_rb_3,f(4),nl(5),sl,'brnn');
hcm_rf_4 = fp_cpu_rl(hcm_rf_3,Gp_rf_4,f(5),nl(6),sl,'frnn');
hcm_rb_4 = fp_cpu_rl(hcm_rb_3,Gp_rb_4,f(5),nl(6),sl,'brnn');

ym_1 = fp_cpu_ll_ow(hcm_rf_4,Gpo_1,f(end));
ym_2 = fp_cpu_ll(hcm_rb_4,Gpo_2,f(end));
ym = ym_1 + ym_2;