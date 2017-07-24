% encoder

[p_mha_1] = update_params_mha(p_mha_1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_ln_1_1] = update_params_ln(p_ln_1_1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);

[p_ff_1_1] = update_params_ll(p_ff_1_1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_ff_1_2] = update_params_ll(p_ff_1_2,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_ln_1_2] = update_params_ln(p_ln_1_2,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);


% decoder
[p_mha_1_dec] = update_params_mha(p_mha_1_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_ln_1_1_dec] = update_params_ln(p_ln_1_1_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);

[p_mha_2_dec] = update_params_mha(p_mha_2_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_ln_1_2_dec] = update_params_ln(p_ln_1_2_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);

[p_ff_1_1_dec] = update_params_ll(p_ff_1_1_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_ff_1_2_dec] = update_params_ll(p_ff_1_2_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[p_ln_1_3_dec] = update_params_ln(p_ln_1_3_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);

[p_ff_1_3_dec] = update_params_ll(p_ff_1_3_dec,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
