% weight initialization
p = []; [Gpi_1] = wt_init_ll(p,gpu_flag,nl(2),nl(1),si,wtdir,wtinit_meth,sgd_type);
p = []; [Gp_rf_1] = wt_init_rl(p,gpu_flag,nl(3),nl(2),si,ri,wtdir,wtinit_meth,sgd_type);
p = []; [Gp_rb_1] = wt_init_rl(p,gpu_flag,nl(3),nl(2),si,ri,wtdir,wtinit_meth,sgd_type);
p = []; [Gp_rf_2] = wt_init_rl(p,gpu_flag,nl(4),nl(3),si,ri,wtdir,wtinit_meth,sgd_type);
p = []; [Gp_rb_2] = wt_init_rl(p,gpu_flag,nl(4),nl(3),si,ri,wtdir,wtinit_meth,sgd_type);
p = []; [Gp_rf_3] = wt_init_rl(p,gpu_flag,nl(5),nl(4),si,ri,wtdir,wtinit_meth,sgd_type);
p = []; [Gp_rb_3] = wt_init_rl(p,gpu_flag,nl(5),nl(4),si,ri,wtdir,wtinit_meth,sgd_type);
p = []; [Gp_rf_4] = wt_init_rl(p,gpu_flag,nl(6),nl(5),si,ri,wtdir,wtinit_meth,sgd_type);
p = []; [Gp_rb_4] = wt_init_rl(p,gpu_flag,nl(6),nl(5),si,ri,wtdir,wtinit_meth,sgd_type);
p = []; [Gpo_1] = wt_init_ll_ow(p,gpu_flag,nl(end),nl(end-1),so,wtdir,wtinit_meth,sgd_type);
p = []; [Gpo_2] = wt_init_ll(p,gpu_flag,nl(end),nl(end-1),so,wtdir,wtinit_meth,sgd_type);

% get full arch_name
get_fullarchname

% train the model
if gpu_flag
    trainrnn_gpu
else
    trainrnn_cpu
end
