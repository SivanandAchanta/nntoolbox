function [p] = update_params_ll_ow(p,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up)
switch sgd_type
    
    case 'sgdcm'
        
        [p.U, p.pdU] = sgdcm(lr, mf, p.gU, p.pdU, p.U);
        
    case 'adadelta'
        
        [p.U, p.pmsgU, p.pmxgU, p.pdU] = adadelta(rho_hp, eps_hp, mf, p.gU, p.pmsgU, p.pmxgU, p.pdU, p.U);
        
    case 'adam'
        
        [p.U,p.pmU,p.pvU]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU,p.pmU,p.pvU,p.U);
        
end



