function [p] = update_params_ll(p,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2)
switch sgd_type
    
    case 'sgdcm'
                
        [p.U, p.pdU] = sgdcm(lr, mf, p.gU, p.pdU, p.U);
        [p.bu, p.pdbu] = sgdcm(lr, mf, p.gbu, p.pdbu, p.bu);
        
    case 'adadelta'
        
        [p.U, p.pmsgU, p.pmsxU, p.pdU] = adadelta(rho_hp, eps_hp, mf, p.gU, p.pmsgU, p.pmsxU, p.pdU, p.U);
        [p.bu, p.pmsgbu, p.pmsxbu, p.pdbu] = adadelta(rho_hp, eps_hp, mf, p.gbu, p.pmsgbu, p.pmsxbu, p.pdbu, p.bu);
        
    case 'adam'
        
        [p.U,p.pmU,p.pvU]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU,p.pmU,p.pvU,p.U);
        [p.bu,p.pmbu,p.pvbu]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gbu,p.pmbu,p.pvbu,p.bu);
        
end



