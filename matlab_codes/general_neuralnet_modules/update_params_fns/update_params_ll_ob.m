function [p] = update_params_ll_ob(p,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up)
switch sgd_type
    
    case 'sgdcm'
        [p.bu, p.pdbu] = sgdcm(lr, mf, p.gbu, p.pdbu, p.bu);
        
    case 'adadelta'
        [p.bu, p.pmsgbu, p.pmsxbu, p.pdbu] = adadelta(rho_hp, eps_hp, mf, p.gbu, p.pmsgbu, p.pmsxbu, p.pdbu, p.bu);
        
    case 'adam'
        [p.bu,p.pmbu,p.pvbu]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gbu,p.pmbu,p.pvbu,p.bu);
        
end



