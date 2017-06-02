function [p] = update_params_slstm(p,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up)

switch sgd_type
    
    case 'sgdcm'
        
        [p.Wz, p.pdWz] = sgdcm(lr, mf, p.gWz, p.pdWz, p.Wz);
        [p.Rz, p.pdRz] = sgdcm(lr, mf, p.gRz, p.pdRz, p.Rz);
        [p.bz, p.pdbz] = sgdcm(lr, mf, p.gbz, p.pdbz, p.bz);
        [p.Wf, p.pdWf] = sgdcm(lr, mf, p.gWf, p.pdWf, p.Wf);
        [p.Rf, p.pdRf] = sgdcm(lr, mf, p.gRf, p.pdRf, p.Rf);
        [p.bf, p.pdbf] = sgdcm(lr, mf, p.gbf, p.pdbf, p.bf);
        
        
    case 'adadelta'
        
        [p.Wz, p.pmsgWz, p.pmsxWz, p.pdWz] = adadelta(rho_hp, eps_hp, mf, p.gWz, p.pmsgWz, p.pmsxWz, p.pdWz, p.Wz);
        [p.Rz, p.pmsgRz, p.pmsxRz, p.pdRz] = adadelta(rho_hp, eps_hp, mf, p.gRz, p.pmsgRz, p.pmsxRz, p.pdRz, p.Rz);
        [p.bz, p.pmsgbz, p.pmsxbz, p.pdbz] = adadelta(rho_hp, eps_hp, mf, p.gbz, p.pmsgbz, p.pmsxbz, p.pdbz, p.bz);
        [p.Wf, p.pmsgWf, p.pmsxWf, p.pdWf] = adadelta(rho_hp, eps_hp, mf, p.gWf, p.pmsgWf, p.pmsxWf, p.pdWf, p.Wf);
        [p.Rf, p.pmsgRf, p.pmsxRf, p.pdRf] = adadelta(rho_hp, eps_hp, mf, p.gRf, p.pmsgRf, p.pmsxRf, p.pdRf, p.Rf);
        [p.bf, p.pmsgbf, p.pmsxbf, p.pdbf] = adadelta(rho_hp, eps_hp, mf, p.gbf, p.pmsgbf, p.pmsxbf, p.pdbf, p.bf);
        
        
    case 'adam'
        
        [p.Wz,p.pmWz,p.pvWz]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gWz,p.pmWz,p.pvWz,p.Wz);
        [p.Rz,p.pmRz,p.pvRz]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gRz,p.pmRz,p.pvRz,p.Rz);
        [p.bz,p.pmbz,p.pvbz]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gbz,p.pmbz,p.pvbz,p.bz);
        [p.Wf,p.pmWf,p.pvWf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gWf,p.pmWf,p.pvWf,p.Wf);
        [p.Rf,p.pmRf,p.pvRf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gRf,p.pmRf,p.pvRf,p.Rf);
        [p.bf,p.pmbf,p.pvbf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gbf,p.pmbf,p.pvbf,p.bf);
        
        
end



        
end



