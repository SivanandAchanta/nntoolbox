function [p] = wt_init_gru(p,gpu_flag,nout,nin,si,ri,fb_init,wtdir,wtinit_meth,sgd_type)



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
        
        Wc = si*randn(nout,nin);
        Rc = ri*randn(nout,nout);
        bc = zeros(nout,1);

    otherwise

        fprintf('Please enter any of the above initialization methods \n');
        return        
end
p.Wz = Wz;
p.Rz = Rz;
p.bz = bz;
p.Wf = Wf;
p.Rf = Rf;
p.bf = bf;
p.Wc = Wc;
p.Rc = Rc;
p.bc = bc;

switch sgd_type
    case 'sgdcm'
        p.pdWz = zeros(size(Wz));
        p.pdRz = zeros(size(Rz));
        p.pdbz = zeros(size(bz));
        p.pdWf = zeros(size(Wf));
        p.pdRf = zeros(size(Rf));
        p.pdbf = zeros(size(bf));
        p.pdWc = zeros(size(Wc));
        p.pdRc = zeros(size(Rc));
        p.pdbc = zeros(size(bc));
        
        
        
    case 'adadelta'
        
        p.pdWz = zeros(size(Wz));
        p.pdRz = zeros(size(Rz));
        p.pdbz = zeros(size(bz));
        p.pdWf = zeros(size(Wf));
        p.pdRf = zeros(size(Rf));
        p.pdbf = zeros(size(bf));
        p.pdWc = zeros(size(Wc));
        p.pdRc = zeros(size(Rc));
        p.pdbc = zeros(size(bc));
        
        
        
        p.pmsgWz = zeros(size(Wz));
        p.pmsgRz = zeros(size(Rz));
        p.pmsgbz = zeros(size(bz));
        p.pmsgWf = zeros(size(Wf));
        p.pmsgRf = zeros(size(Rf));
        p.pmsgbf = zeros(size(bf));
        p.pmsgWc = zeros(size(Wc));
        p.pmsgRc = zeros(size(Rc));
        p.pmsgbc = zeros(size(bc));
        
        
        p.pmsxWz = zeros(size(Wz));
        p.pmsxRz = zeros(size(Rz));
        p.pmsxbz = zeros(size(bz));
        p.pmsxWf = zeros(size(Wf));
        p.pmsxRf = zeros(size(Rf));
        p.pmsxbf = zeros(size(bf));
        p.pmsxWc = zeros(size(Wc));
        p.pmsxRc = zeros(size(Rc));
        p.pmsxbc = zeros(size(bc));
        
        
    case 'adam'
        
        p.pmWz = zeros(size(Wz));
        p.pmRz = zeros(size(Rz));
        p.pmbz = zeros(size(bz));
        p.pmWf = zeros(size(Wf));
        p.pmRf = zeros(size(Rf));
        p.pmbf = zeros(size(bf));
        p.pmWc = zeros(size(Wc));
        p.pmRc = zeros(size(Rc));
        p.pmbc = zeros(size(bc));
        
        
        p.pvWz = zeros(size(Wz));
        p.pvRz = zeros(size(Rz));
        p.pvbz = zeros(size(bz));
        p.pvWf = zeros(size(Wf));
        p.pvRf = zeros(size(Rf));
        p.pvbf = zeros(size(bf));
        p.pvWc = zeros(size(Wc));
        p.pvRc = zeros(size(Rc));
        p.pvbc = zeros(size(bc));
        
        
        
end


end
