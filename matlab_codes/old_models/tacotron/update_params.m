% encoder
[Em] = update_params_embedding(Em,ix_vec,Eb,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_f_1] = update_params_ll(p_f_1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_f_2] = update_params_ll(p_f_2,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_h_1] = update_params_hl(p_h_1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_h_2] = update_params_hl(p_h_2,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_h_3] = update_params_hl(p_h_3,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_h_4] = update_params_hl(p_h_4,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_lf_1] = update_params_lstm(p_lf_1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_lb_1] = update_params_lstm(p_lb_1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);

% attention
[p_f_3_0] = update_params_ll_ob(p_f_3_0,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_f_3_1] = update_params_ll_ow(p_f_3_1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_f_3_2] = update_params_ll_ow(p_f_3_2,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);

% decoder
[p_f_1_dec] = update_params_ll(p_f_1_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_f_2_dec] = update_params_ll(p_f_2_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_lf_1_dec] = update_params_lstm(p_lf_1_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_lf_2_dec] = update_params_lstm(p_lf_2_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_lf_3_dec] = update_params_lstm(p_lf_3_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_f_4_1_dec] = update_params_ll(p_f_4_1_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_f_4_2_dec] = update_params_ll(p_f_4_2_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
