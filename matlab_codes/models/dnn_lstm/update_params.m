
switch sgd_type
    
    case 'sgdcm'
        
        
        [Wz,pdWz]     = sgdcm(lr,mf,gWz,pdWz,Wz);
        [Rz,pdRz]     = sgdcm(lr,mf,gRz,pdRz,Rz);
        [bz,pdbz]     = sgdcm(lr,mf,gbz,pdbz,bz);
        [Wi,pdWi]     = sgdcm(lr,mf,gWi,pdWi,Wi);
        [Ri,pdRi]     = sgdcm(lr,mf,gRi,pdRi,Ri);
        [bi,pdbi]     = sgdcm(lr,mf,gbi,pdbi,bi);
        [pi,pdpi]     = sgdcm(lr,mf,gpi,pdpi,pi);
        [Wf,pdWf]     = sgdcm(lr,mf,gWf,pdWf,Wf);
        [Rf,pdRf]     = sgdcm(lr,mf,gRf,pdRf,Rf);
        [bf,pdbf]     = sgdcm(lr,mf,gbf,pdbf,bf);
        [pf,pdpf]     = sgdcm(lr,mf,gpf,pdpf,pf);
        [Wo,pdWo]     = sgdcm(lr,mf,gWo,pdWo,Wo);
        [Ro,pdRo]     = sgdcm(lr,mf,gRo,pdRo,Ro);
        [bo,pdbo]     = sgdcm(lr,mf,gbo,pdbo,bo);
        [po,pdpo]     = sgdcm(lr,mf,gpo,pdpo,po);
        [U,pdU]       = sgdcm(lr,mf,gU,pdU,U);
        [bu,pdbu]     = sgdcm(lr,mf,gbu,pdbu,bu);
        
        
    case 'adadelta'
        
        
        [Wz,pmsgWz,pmsxWz,pdWz]     = adadelta(rho,eps_hp,mf,gWz,pmsgWz,pmsxWz,pdWz,Wz);
        [Rz,pmsgRz,pmsxRz,pdRz]     = adadelta(rho,eps_hp,mf,gRz,pmsgRz,pmsxRz,pdRz,Rz);
        [bz,pmsgbz,pmsxbz,pdbz]     = adadelta(rho,eps_hp,mf,gbz,pmsgbz,pmsxbz,pdbz,bz);
        [Wi,pmsgWi,pmsxWi,pdWi]     = adadelta(rho,eps_hp,mf,gWi,pmsgWi,pmsxWi,pdWi,Wi);
        [Ri,pmsgRi,pmsxRi,pdRi]     = adadelta(rho,eps_hp,mf,gRi,pmsgRi,pmsxRi,pdRi,Ri);
        [bi,pmsgbi,pmsxbi,pdbi]     = adadelta(rho,eps_hp,mf,gbi,pmsgbi,pmsxbi,pdbi,bi);
        [pi,pmsgpi,pmsxpi,pdpi]     = adadelta(rho,eps_hp,mf,gpi,pmsgpi,pmsxpi,pdpi,pi);
        [Wf,pmsgWf,pmsxWf,pdWf]     = adadelta(rho,eps_hp,mf,gWf,pmsgWf,pmsxWf,pdWf,Wf);
        [Rf,pmsgRf,pmsxRf,pdRf]     = adadelta(rho,eps_hp,mf,gRf,pmsgRf,pmsxRf,pdRf,Rf);
        [bf,pmsgbf,pmsxbf,pdbf]     = adadelta(rho,eps_hp,mf,gbf,pmsgbf,pmsxbf,pdbf,bf);
        [pf,pmsgpf,pmsxpf,pdpf]     = adadelta(rho,eps_hp,mf,gpf,pmsgpf,pmsxpf,pdpf,pf);
        [Wo,pmsgWo,pmsxWo,pdWo]     = adadelta(rho,eps_hp,mf,gWo,pmsgWo,pmsxWo,pdWo,Wo);
        [Ro,pmsgRo,pmsxRo,pdRo]     = adadelta(rho,eps_hp,mf,gRo,pmsgRo,pmsxRo,pdRo,Ro);
        [bo,pmsgbo,pmsxbo,pdbo]     = adadelta(rho,eps_hp,mf,gbo,pmsgbo,pmsxbo,pdbo,bo);
        [po,pmsgpo,pmsxpo,pdpo]     = adadelta(rho,eps_hp,mf,gpo,pmsgpo,pmsxpo,pdpo,po);
        [U,pmsgU,pmsxU,pdU]         = adadelta(rho,eps_hp,mf,gU,pmsgU,pmsxU,pdU,U);
        [bu,pmsgbu,pmsxbu,pdbu]     = adadelta(rho,eps_hp,mf,gbu,pmsgbu,pmsxbu,pdbu,bu);
        
    case 'adam'
        
        [Wz,pmWz,pvWz]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gWz,pmWz,pvWz,Wz);
        [Rz,pmRz,pvRz]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gRz,pmRz,pvRz,Rz);
        [bz,pmbz,pvbz]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gbz,pmbz,pvbz,bz);
        [Wi,pmWi,pvWi]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gWi,pmWi,pvWi,Wi);
        [Ri,pmRi,pvRi]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gRi,pmRi,pvRi,Ri);
        [bi,pmbi,pvbi]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gbi,pmbi,pvbi,bi);
        [pi,pmpi,pvpi]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gpi,pmpi,pvpi,pi);
        [Wf,pmWf,pvWf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gWf,pmWf,pvWf,Wf);
        [Rf,pmRf,pvRf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gRf,pmRf,pvRf,Rf);
        [bf,pmbf,pvbf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gbf,pmbf,pvbf,bf);
        [pf,pmpf,pvpf]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gpf,pmpf,pvpf,pf);
        [Wo,pmWo,pvWo]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gWo,pmWo,pvWo,Wo);
        [Ro,pmRo,pvRo]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gRo,pmRo,pvRo,Ro);
        [bo,pmbo,pvbo]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gbo,pmbo,pvbo,bo);
        [po,pmpo,pvpo]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gpo,pmpo,pvpo,po);
        [U,pmU,pvU]         = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gU,pmU,pvU,U);
        [bu,pmbu,pvbu]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gbu,pmbu,pvbu,bu);
end



if dnn_flag    

    switch sgd_type
        
        case 'sgdcm'            
            [W, pdW] = sgdcm(lr, mf, gW, pdW, W);
            [b, pdb] = sgdcm(lr, mf, gb, pdb, b);
            
        case 'adadelta'
            [W, pmsgW, pmsxW, pdW] = adadelta(rho, eps_hp, mf, gW, pmsgW, pmsxW, pdW, W);
            [b, pmsgb, pmsxb, pdb] = adadelta(rho, eps_hp, mf, gb, pmsgb, pmsxb, pdb, b);
            
        case 'adam'
            [W,pmW,pvW]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gW,pmW,pvW,W);
            [b,pmb,pvb]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gb,pmb,pvb,b);            
            
    end
end

    
