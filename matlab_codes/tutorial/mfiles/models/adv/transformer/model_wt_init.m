% weight initialization

p = []; [p_mha_1] = wt_init_mha(p,gpu_flag,d_model,num_heads,wtdir,wtinit_meth,sgd_type)
p = []; [p_ln_1_1] = wt_init_ln(p,gpu_flag,d_model,wtdir,wtinit_meth,sgd_type)

p = []; [p_ff_1_1] = wt_init_ll(p,gpu_flag,d_ff,d_model,si,wtdir,wtinit_meth,sgd_type);
p = []; [p_ff_1_2] = wt_init_ll(p,gpu_flag,d_model,d_ff,si,wtdir,wtinit_meth,sgd_type);
p = []; [p_ln_1_2] = wt_init_ln(p,gpu_flag,d_model,wtdir,wtinit_meth,sgd_type)


p = []; [p_mha_1_dec] = wt_init_mha(p,gpu_flag,d_model,num_heads,wtdir,wtinit_meth,sgd_type)
p = []; [p_ln_1_1_dec] = wt_init_ln(p,gpu_flag,d_model,wtdir,wtinit_meth,sgd_type)

p = []; [p_mha_2_dec] = wt_init_mha(p,gpu_flag,d_model,num_heads,wtdir,wtinit_meth,sgd_type)
p = []; [p_ln_1_2_dec] = wt_init_ln(p,gpu_flag,d_model,wtdir,wtinit_meth,sgd_type)

p = []; [p_ff_1_1_dec] = wt_init_ll(p,gpu_flag,d_ff,d_model,si,wtdir,wtinit_meth,sgd_type);
p = []; [p_ff_1_2_dec] = wt_init_ll(p,gpu_flag,d_model,d_ff,si,wtdir,wtinit_meth,sgd_type);
p = []; [p_ln_1_3_dec] = wt_init_ln(p,gpu_flag,d_model,wtdir,wtinit_meth,sgd_type)

p = []; [p_ff_1_3_dec] = wt_init_ll(p,gpu_flag,nl(end),d_model,si,wtdir,wtinit_meth,sgd_type);
