function [p] = update_params_hl(p,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up)

switch sgd_type
    
    case 'sgdcm'
        
        [p.W, p.pdW] = sgdcm(lr, mf, p.gW, p.pdW, p.W);
        [p.Wt, p.pdWt] = sgdcm(lr, mf, p.gWt, p.pdWt, p.Wt);
        [p.b, p.pdb] = sgdcm(lr, mf, p.gb, p.pdb, p.b);
        [p.bt, p.pdbt] = sgdcm(lr, mf, p.gbt, p.pdbt, p.bt);
        
        
    case 'adadelta'
        
        [p.W, p.pmsgW, p.pmsxW, p.pdW] = adadelta(rho_hp, eps_hp, mf, p.gW, p.pmsgW, p.pmsxW, p.pdW, p.W);
        [p.Wt, p.pmsgWt, p.pmsxWt, p.pdWt] = adadelta(rho_hp, eps_hp, mf, p.gWt, p.pmsgWt, p.pmsxWt, p.pdWt, p.Wt);
        [p.b, p.pmsgb, p.pmsxb, p.pdb] = adadelta(rho_hp, eps_hp, mf, p.gb, p.pmsgb, p.pmsxb, p.pdb, p.b);
        [p.bt, p.pmsgbt, p.pmsxbt, p.pdbt] = adadelta(rho_hp, eps_hp, mf, p.gbt, p.pmsgbt, p.pmsxbt, p.pdbt, p.bt);
        
        
    case 'adam'
        
        [p.W,p.pmW,p.pvW]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gW,p.pmW,p.pvW,p.W);
        [p.Wt,p.pmWt,p.pvWt]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gWt,p.pmWt,p.pvWt,p.Wt);
        [p.b,p.pmb,p.pvb]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gb,p.pmb,p.pvb,p.b);
        [p.bt,p.pmbt,p.pvbt]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gbt,p.pmbt,p.pvbt,p.bt);
        
        
end



