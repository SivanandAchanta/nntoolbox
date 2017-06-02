function [p] = update_params_rnn(p,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up)
switch sgd_type
    
    case 'sgdcm'
        
        [p.Wi, p.pdWi] = sgdcm(lr, mf, p.gWi, p.pdWi, p.Wi);
        [p.Wfr, p.pdWfr] = sgdcm(lr, mf, p.gWfr, p.pdWfr, p.Wfr);
        [p.bh, p.pdbh] = sgdcm(lr, mf, p.gbh, p.pdbh, p.bh);
        
        
    case 'adadelta'
        
        [p.Wi, p.pmsgWi, p.pmsxWi, p.pdWi] = adadelta(rho_hp, eps_hp, mf, p.gWi, p.pmsgWi, p.pmsxWi, p.pdWi, p.Wi);
        [p.Wfr, p.pmsgWfr, p.pmsxWfr, p.pdWfr] = adadelta(rho_hp, eps_hp, mf, p.gWfr, p.pmsgWfr, p.pmsxWfr, p.pdWfr, p.Wfr);
        [p.bh, p.pmsgbh, p.pmsxbh, p.pdbh] = adadelta(rho_hp, eps_hp, mf, p.gbh, p.pmsgbh, p.pmsxbh, p.pdbh, p.bh);
        
        
    case 'adam'
        
        [p.Wi,p.pmWi,p.pvWi]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gWi,p.pmWi,p.pvWi,p.Wi);
        [p.Wfr,p.pmWfr,p.pvWfr]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gWfr,p.pmWfr,p.pvWfr,p.Wfr);
        [p.bh,p.pmbh,p.pvbh]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gbh,p.pmbh,p.pvbh,p.bh);
        
        
end



