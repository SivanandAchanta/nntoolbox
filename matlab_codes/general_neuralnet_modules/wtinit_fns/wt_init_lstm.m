function [p] = wt_init_lstm(p,nout,nin,si,fb_init,wtdir,wtinit_meth,sgd_type)



switch wtinit_meth
    
    case 'ui'
        
        si = sqrt(1/(nout*nin));
        Wf = 2*si*rand(nout,nin) - si;
        si = (1/nout);
        Rf = 2*si*rand(nout,nout) - si;
        bf = fb_init*ones(nout,1);
        
    case 'gi'
        Wz = si*randn(nout,nin);
        Wi = si*randn(nout,nin);
        Wf = si*randn(nout,nin);
        Wo = si*randn(nout,nin);
        
        Rz = si*randn(nout,nout);
        Ri = si*randn(nout,nout);
        Rf = si*randn(nout,nout);
        Ro = si*randn(nout,nout);
        
        bz = zeros(nout,1);
        bi = zeros(nout,1);
        bf = fb_init*ones(nout,1);
        bo = zeros(nout,1);
        
        pi = zeros(nout,1);
        pf = zeros(nout,1);
        po = zeros(nout,1);
        
end

p.Wz = Wz;
p.Rz = Rz;
p.bz = bz;
p.Wf = Wf;
p.Rf = Rf;
p.bf = bf;
p.Wi = Wi;
p.Ri = Ri;
p.bi = bi;
p.Wo = Wo;
p.Ro = Ro;
p.bo = bo;
p.pi = pi;
p.pf = pf;
p.po = po;

switch sgd_type
    case 'sgdcm'
        p.pdWz = zeros(size(Wz));
        p.pdRz = zeros(size(Rz));
        p.pdbz = zeros(size(bz));
        p.pdWf = zeros(size(Wf));
        p.pdRf = zeros(size(Rf));
        p.pdbf = zeros(size(bf));
        p.pdWi = zeros(size(Wi));
        p.pdRi = zeros(size(Ri));
        p.pdbi = zeros(size(bi));
        p.pdWo = zeros(size(Wo));
        p.pdRo = zeros(size(Ro));
        p.pdbo = zeros(size(bo));
        p.pdpi = zeros(size(pi));
        p.pdpf = zeros(size(pf));
        p.pdpo = zeros(size(po));
        
    case 'adadelta'
        
        p.pdWz = zeros(size(Wz));
        p.pdRz = zeros(size(Rz));
        p.pdbz = zeros(size(bz));
        p.pdWf = zeros(size(Wf));
        p.pdRf = zeros(size(Rf));
        p.pdbf = zeros(size(bf));
        p.pdWi = zeros(size(Wi));
        p.pdRi = zeros(size(Ri));
        p.pdbi = zeros(size(bi));
        p.pdWo = zeros(size(Wo));
        p.pdRo = zeros(size(Ro));
        p.pdbo = zeros(size(bo));
        p.pdpi = zeros(size(pi));
        p.pdpf = zeros(size(pf));
        p.pdpo = zeros(size(po));
        
        p.pmsgWz = zeros(size(Wz));
        p.pmsgRz = zeros(size(Rz));
        p.pmsgbz = zeros(size(bz));
        p.pmsgWf = zeros(size(Wf));
        p.pmsgRf = zeros(size(Rf));
        p.pmsgbf = zeros(size(bf));
        p.pmsgWi = zeros(size(Wi));
        p.pmsgRi = zeros(size(Ri));
        p.pmsgbi = zeros(size(bi));
        p.pmsgWo = zeros(size(Wo));
        p.pmsgRo = zeros(size(Ro));
        p.pmsgbo = zeros(size(bo));
        p.pmsgpi = zeros(size(pi));
        p.pmsgpf = zeros(size(pf));
        p.pmsgpo = zeros(size(po));
        
        p.pmsxWz = zeros(size(Wz));
        p.pmsxRz = zeros(size(Rz));
        p.pmsxbz = zeros(size(bz));
        p.pmsxWf = zeros(size(Wf));
        p.pmsxRf = zeros(size(Rf));
        p.pmsxbf = zeros(size(bf));
        p.pmsxWi = zeros(size(Wi));
        p.pmsxRi = zeros(size(Ri));
        p.pmsxbi = zeros(size(bi));
        p.pmsxWo = zeros(size(Wo));
        p.pmsxRo = zeros(size(Ro));
        p.pmsxbo = zeros(size(bo));
        p.pmsxpi = zeros(size(pi));
        p.pmsxpf = zeros(size(pf));
        p.pmsxpo = zeros(size(po));
        
    case 'adam'
        
        p.pmWz = zeros(size(Wz));
        p.pmRz = zeros(size(Rz));
        p.pmbz = zeros(size(bz));
        p.pmWf = zeros(size(Wf));
        p.pmRf = zeros(size(Rf));
        p.pmbf = zeros(size(bf));
        p.pmWi = zeros(size(Wi));
        p.pmRi = zeros(size(Ri));
        p.pmbi = zeros(size(bi));
        p.pmWo = zeros(size(Wo));
        p.pmRo = zeros(size(Ro));
        p.pmbo = zeros(size(bo));
        p.pmpi = zeros(size(pi));
        p.pmpf = zeros(size(pf));
        p.pmpo = zeros(size(po));
        
        p.pvWz = zeros(size(Wz));
        p.pvRz = zeros(size(Rz));
        p.pvbz = zeros(size(bz));
        p.pvWf = zeros(size(Wf));
        p.pvRf = zeros(size(Rf));
        p.pvbf = zeros(size(bf));
        p.pvWi = zeros(size(Wi));
        p.pvRi = zeros(size(Ri));
        p.pvbi = zeros(size(bi));
        p.pvWo = zeros(size(Wo));
        p.pvRo = zeros(size(Ro));
        p.pvbo = zeros(size(bo));
        p.pvpi = zeros(size(pi));
        p.pvpf = zeros(size(pf));
        p.pvpo = zeros(size(po));
end


end
