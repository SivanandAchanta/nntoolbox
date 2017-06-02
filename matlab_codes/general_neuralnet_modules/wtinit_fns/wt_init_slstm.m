function [p] = wt_init_slstm(p,gpu_flag,nout,nin,si,ri,fb_init,wtdir,wtinit_meth,sgd_type)



switch wtinit_meth
    
    case 'ui'
        
        si = sqrt(1/(nout*nin));
        Wf = 2*si*rand(nout,nin) - si;
        ri = (1/nout);
        Rf = 2*ri*rand(nout,nout) - ri;
        bf = fb_init*ones(nout,1);
        
    case 'gi'
        Wz = si*randn(nout,nin);
        Rz = ri*randn(nout,nout);
        bz = zeros(nout,1);
        
        Wf = si*randn(nout,nin);
        Rf = ri*randn(nout,nout);
        bf = fb_init*ones(nout,1);
        
end
p.Wz = Wz;
p.Rz = Rz;
p.bz = bz;
p.Wf = Wf;
p.Rf = Rf;
p.bf = bf;

switch sgd_type
    case 'sgdcm'
        p.pdWz = zeros(size(Wz));
        p.pdRz = zeros(size(Rz));
        p.pdbz = zeros(size(bz));
        p.pdWf = zeros(size(Wf));
        p.pdRf = zeros(size(Rf));
        p.pdbf = zeros(size(bf));
        
        
        
    case 'adadelta'
        
        p.pdWz = zeros(size(Wz));
        p.pdRz = zeros(size(Rz));
        p.pdbz = zeros(size(bz));
        p.pdWf = zeros(size(Wf));
        p.pdRf = zeros(size(Rf));
        p.pdbf = zeros(size(bf));
        
        
        
        p.pmsgWz = zeros(size(Wz));
        p.pmsgRz = zeros(size(Rz));
        p.pmsgbz = zeros(size(bz));
        p.pmsgWf = zeros(size(Wf));
        p.pmsgRf = zeros(size(Rf));
        p.pmsgbf = zeros(size(bf));
        
        
        p.pmsxWz = zeros(size(Wz));
        p.pmsxRz = zeros(size(Rz));
        p.pmsxbz = zeros(size(bz));
        p.pmsxWf = zeros(size(Wf));
        p.pmsxRf = zeros(size(Rf));
        p.pmsxbf = zeros(size(bf));
        
        
    case 'adam'
        
        p.pmWz = zeros(size(Wz));
        p.pmRz = zeros(size(Rz));
        p.pmbz = zeros(size(bz));
        p.pmWf = zeros(size(Wf));
        p.pmRf = zeros(size(Rf));
        p.pmbf = zeros(size(bf));
        
        
        p.pvWz = zeros(size(Wz));
        p.pvRz = zeros(size(Rz));
        p.pvbz = zeros(size(bz));
        p.pvWf = zeros(size(Wf));
        p.pvRf = zeros(size(Rf));
        p.pvbf = zeros(size(bf));
        
        
        
end


end
