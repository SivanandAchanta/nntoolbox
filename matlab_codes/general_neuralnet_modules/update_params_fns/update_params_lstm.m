function [p] = update_params_lstm(p,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up)

switch sgd_type
    
    case 'sgdcm'
        
        [p.Wz,p.pdWz]     = sgdcm(lr,mf,p.gWz,p.pdWz,p.Wz);
        [p.Rz,p.pdRz]     = sgdcm(lr,mf,p.gRz,p.pdRz,p.Rz);
        [p.bz,p.pdbz]     = sgdcm(lr,mf,p.gbz,p.pdbz,p.bz);
        [p.Wi,p.pdWi]     = sgdcm(lr,mf,p.gWi,p.pdWi,p.Wi);
        [p.Ri,p.pdRi]     = sgdcm(lr,mf,p.gRi,p.pdRi,p.Ri);
        [p.bi,p.pdbi]     = sgdcm(lr,mf,p.gbi,p.pdbi,p.bi);
        [p.pi,p.pdpi]     = sgdcm(lr,mf,p.gpi,p.pdpi,p.pi);
        [p.Wf,p.pdWf]     = sgdcm(lr,mf,p.gWf,p.pdWf,p.Wf);
        [p.Rf,p.pdRf]     = sgdcm(lr,mf,p.gRf,p.pdRf,p.Rf);
        [p.bf,p.pdbf]     = sgdcm(lr,mf,p.gbf,p.pdbf,p.bf);
        [p.pf,p.pdpf]     = sgdcm(lr,mf,p.gpf,p.pdpf,p.pf);
        [p.Wo,p.pdWo]     = sgdcm(lr,mf,p.gWo,p.pdWo,p.Wo);
        [p.Ro,p.pdRo]     = sgdcm(lr,mf,p.gRo,p.pdRo,p.Ro);
        [p.bo,p.pdbo]     = sgdcm(lr,mf,p.gbo,p.pdbo,p.bo);
        [p.po,p.pdpo]     = sgdcm(lr,mf,p.gpo,p.pdpo,p.po);
        
        
    case 'adadelta'
        
        [p.Wz,p.pmsgWz,p.pmsxWz,p.pdWz]     = adadelta(rho_hp,eps_hp,mf,p.gWz,p.pmsgWz,p.pmsxWz,p.pdWz,p.Wz);
        [p.Rz,p.pmsgRz,p.pmsxRz,p.pdRz]     = adadelta(rho_hp,eps_hp,mf,p.gRz,p.pmsgRz,p.pmsxRz,p.pdRz,p.Rz);
        [p.bz,p.pmsgbz,p.pmsxbz,p.pdbz]     = adadelta(rho_hp,eps_hp,mf,p.gbz,p.pmsgbz,p.pmsxbz,p.pdbz,p.bz);
        [p.Wi,p.pmsgWi,p.pmsxWi,p.pdWi]     = adadelta(rho_hp,eps_hp,mf,p.gWi,p.pmsgWi,p.pmsxWi,p.pdWi,p.Wi);
        [p.Ri,p.pmsgRi,p.pmsxRi,p.pdRi]     = adadelta(rho_hp,eps_hp,mf,p.gRi,p.pmsgRi,p.pmsxRi,p.pdRi,p.Ri);
        [p.bi,p.pmsgbi,p.pmsxbi,p.pdbi]     = adadelta(rho_hp,eps_hp,mf,p.gbi,p.pmsgbi,p.pmsxbi,p.pdbi,p.bi);
        [p.pi,p.pmsgpi,p.pmsxpi,p.pdpi]     = adadelta(rho_hp,eps_hp,mf,p.gpi,p.pmsgpi,p.pmsxpi,p.pdpi,p.pi);
        [p.Wf,p.pmsgWf,p.pmsxWf,p.pdWf]     = adadelta(rho_hp,eps_hp,mf,p.gWf,p.pmsgWf,p.pmsxWf,p.pdWf,p.Wf);
        [p.Rf,p.pmsgRf,p.pmsxRf,p.pdRf]     = adadelta(rho_hp,eps_hp,mf,p.gRf,p.pmsgRf,p.pmsxRf,p.pdRf,p.Rf);
        [p.bf,p.pmsgbf,p.pmsxbf,p.pdbf]     = adadelta(rho_hp,eps_hp,mf,p.gbf,p.pmsgbf,p.pmsxbf,p.pdbf,p.bf);
        [p.pf,p.pmsgpf,p.pmsxpf,p.pdpf]     = adadelta(rho_hp,eps_hp,mf,p.gpf,p.pmsgpf,p.pmsxpf,p.pdpf,p.pf);
        [p.Wo,p.pmsgWo,p.pmsxWo,p.pdWo]     = adadelta(rho_hp,eps_hp,mf,p.gWo,p.pmsgWo,p.pmsxWo,p.pdWo,p.Wo);
        [p.Ro,p.pmsgRo,p.pmsxRo,p.pdRo]     = adadelta(rho_hp,eps_hp,mf,p.gRo,p.pmsgRo,p.pmsxRo,p.pdRo,p.Ro);
        [p.bo,p.pmsgbo,p.pmsxbo,p.pdbo]     = adadelta(rho_hp,eps_hp,mf,p.gbo,p.pmsgbo,p.pmsxbo,p.pdbo,p.bo);
        [p.po,p.pmsgpo,p.pmsxpo,p.pdpo]     = adadelta(rho_hp,eps_hp,mf,p.gpo,p.pmsgpo,p.pmsxpo,p.pdpo,p.po);
        
    case 'adam'
        
        [p.Wz,p.pmWz,p.pvWz]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gWz,p.pmWz,p.pvWz,p.Wz);
        [p.Rz,p.pmRz,p.pvRz]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gRz,p.pmRz,p.pvRz,p.Rz);
        [p.bz,p.pmbz,p.pvbz]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gbz,p.pmbz,p.pvbz,p.bz);
        [p.Wi,p.pmWi,p.pvWi]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gWi,p.pmWi,p.pvWi,p.Wi);
        [p.Ri,p.pmRi,p.pvRi]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gRi,p.pmRi,p.pvRi,p.Ri);
        [p.bi,p.pmbi,p.pvbi]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gbi,p.pmbi,p.pvbi,p.bi);
        [p.pi,p.pmpi,p.pvpi]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gpi,p.pmpi,p.pvpi,p.pi);
        [p.Wf,p.pmWf,p.pvWf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gWf,p.pmWf,p.pvWf,p.Wf);
        [p.Rf,p.pmRf,p.pvRf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gRf,p.pmRf,p.pvRf,p.Rf);
        [p.bf,p.pmbf,p.pvbf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gbf,p.pmbf,p.pvbf,p.bf);
        [p.pf,p.pmpf,p.pvpf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gpf,p.pmpf,p.pvpf,p.pf);
        [p.Wo,p.pmWo,p.pvWo]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gWo,p.pmWo,p.pvWo,p.Wo);
        [p.Ro,p.pmRo,p.pvRo]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gRo,p.pmRo,p.pvRo,p.Ro);
        [p.bo,p.pmbo,p.pvbo]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gbo,p.pmbo,p.pvbo,p.bo);
        [p.po,p.pmpo,p.pvpo]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gpo,p.pmpo,p.pvpo,p.po);
        
end




