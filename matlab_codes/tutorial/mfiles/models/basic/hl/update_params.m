% Update Params using Appropriate SGD Method
[Gpi1] = update_params_hl(Gpi1,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[Gpi2] = update_params_hl(Gpi2,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);
[Gpo] = update_params_ll(Gpo,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up);

