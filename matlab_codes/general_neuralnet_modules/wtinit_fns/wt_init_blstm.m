function [p] = wt_init_blstm(p,nout,nin,si,fb_init,wtdir,wtinit_meth,sgd_type)

Wzf = si*randn(nout,nin);
Wif = si*randn(nout,nin);
Wff = si*randn(nout,nin);
Wof = si*randn(nout,nin);

Rzf = si*randn(nout,nout);
Rif = si*randn(nout,nout);
Rff = si*randn(nout,nout);
Rof = si*randn(nout,nout);

bzf = zeros(nout,1);
bif = zeros(nout,1);
bff = fb_init*ones(nout,1);
bof = zeros(nout,1);

pif = zeros(nout,1);
pff = zeros(nout,1);
pof = zeros(nout,1);

Wzb = si*randn(nout,nin);
Wib = si*randn(nout,nin);
Wfb = si*randn(nout,nin);
Wob = si*randn(nout,nin);

Rzb = si*randn(nout,nout);
Rib = si*randn(nout,nout);
Rfb = si*randn(nout,nout);
Rob = si*randn(nout,nout);

bzb = zeros(nout,1);
bib = zeros(nout,1);
bfb = fb_init*ones(nout,1);
bob = zeros(nout,1);

pib = zeros(nout,1);
pfb = zeros(nout,1);
pob = zeros(nout,1);

switch sgd_type
    case 'sgdcm'
        pdWzf = zeros(size(Wzf));
        pdWif = zeros(size(Wif));
        pdWff = zeros(size(Wff));
        pdWof = zeros(size(Wof));
        
        pdRzf = zeros(size(Rzf));
        pdRif = zeros(size(Rif));
        pdRff = zeros(size(Rff));
        pdRof = zeros(size(Rof));
        
        pdbzf = zeros(size(bzf));
        pdbif = zeros(size(bif));
        pdbff = zeros(size(bff));
        pdbof = zeros(size(bof));
        
        pdpif = zeros(size(pif));
        pdpff = zeros(size(pff));
        pdpof = zeros(size(pof));
        
        pdUf  = zeros(size(Uf));
        
        
        pdWzb = zeros(size(Wzb));
        pdWib = zeros(size(Wib));
        pdWfb = zeros(size(Wfb));
        pdWob = zeros(size(Wob));
        
        pdRzb = zeros(size(Rzb));
        pdRib = zeros(size(Rib));
        pdRfb = zeros(size(Rfb));
        pdRob = zeros(size(Rob));
        
        pdbzb = zeros(size(bzb));
        pdbib = zeros(size(bib));
        pdbfb = zeros(size(bfb));
        pdbob = zeros(size(bob));
        
        pdpib = zeros(size(pib));
        pdpfb = zeros(size(pfb));
        pdpob = zeros(size(pob));
        
        pdUb  = zeros(size(Ub));
        
        pdbu = zeros(size(bu));
        
        
    case 'adadelta'
        
        pdWzf = zeros(size(Wzf));
        pdWif = zeros(size(Wif));
        pdWff = zeros(size(Wff));
        pdWof = zeros(size(Wof));
        
        pdRzf = zeros(size(Rzf));
        pdRif = zeros(size(Rif));
        pdRff = zeros(size(Rff));
        pdRof = zeros(size(Rof));
        
        pdbzf = zeros(size(bzf));
        pdbif = zeros(size(bif));
        pdbff = zeros(size(bff));
        pdbof = zeros(size(bof));
        
        pdpif = zeros(size(pif));
        pdpff = zeros(size(pff));
        pdpof = zeros(size(pof));
        
        pdUf  = zeros(size(Uf));
        pdbu = zeros(size(bu));
        
        
        pmsgWzf = zeros(size(Wzf));
        pmsgWif = zeros(size(Wif));
        pmsgWff = zeros(size(Wff));
        pmsgWof = zeros(size(Wof));
        
        pmsgRzf = zeros(size(Rzf));
        pmsgRif = zeros(size(Rif));
        pmsgRff = zeros(size(Rff));
        pmsgRof = zeros(size(Rof));
        
        pmsgbzf = zeros(size(bzf));
        pmsgbif = zeros(size(bif));
        pmsgbff = zeros(size(bff));
        pmsgbof = zeros(size(bof));
        
        pmsgpif = zeros(size(pif));
        pmsgpff = zeros(size(pff));
        pmsgpof = zeros(size(pof));
        
        pmsgUf  = zeros(size(Uf));
        pmsgbu = zeros(size(bu));
        
        pmsxWzf = zeros(size(Wzf));
        pmsxWif = zeros(size(Wif));
        pmsxWff = zeros(size(Wff));
        pmsxWof = zeros(size(Wof));
        
        pmsxRzf = zeros(size(Rzf));
        pmsxRif = zeros(size(Rif));
        pmsxRff = zeros(size(Rff));
        pmsxRof = zeros(size(Rof));
        
        pmsxbzf = zeros(size(bzf));
        pmsxbif = zeros(size(bif));
        pmsxbff = zeros(size(bff));
        pmsxbof = zeros(size(bof));
        
        pmsxpif = zeros(size(pif));
        pmsxpff = zeros(size(pff));
        pmsxpof = zeros(size(pof));
        
        pmsxUf  = zeros(size(Uf));
        pmsxbu = zeros(size(bu));
        
        
        pdWzb = zeros(size(Wzb));
        pdWib = zeros(size(Wib));
        pdWfb = zeros(size(Wfb));
        pdWob = zeros(size(Wob));
        
        pdRzb = zeros(size(Rzb));
        pdRib = zeros(size(Rib));
        pdRfb = zeros(size(Rfb));
        pdRob = zeros(size(Rob));
        
        pdbzb = zeros(size(bzb));
        pdbib = zeros(size(bib));
        pdbfb = zeros(size(bfb));
        pdbob = zeros(size(bob));
        
        pdpib = zeros(size(pib));
        pdpfb = zeros(size(pfb));
        pdpob = zeros(size(pob));
        
        pdUb  = zeros(size(Ub));
        
        
        pmsgWzb = zeros(size(Wzb));
        pmsgWib = zeros(size(Wib));
        pmsgWfb = zeros(size(Wfb));
        pmsgWob = zeros(size(Wob));
        
        pmsgRzb = zeros(size(Rzb));
        pmsgRib = zeros(size(Rib));
        pmsgRfb = zeros(size(Rfb));
        pmsgRob = zeros(size(Rob));
        
        pmsgbzb = zeros(size(bzb));
        pmsgbib = zeros(size(bib));
        pmsgbfb = zeros(size(bfb));
        pmsgbob = zeros(size(bob));
        
        pmsgpib = zeros(size(pib));
        pmsgpfb = zeros(size(pfb));
        pmsgpob = zeros(size(pob));
        
        pmsgUb  = zeros(size(Ub));
        
        pmsxWzb = zeros(size(Wzb));
        pmsxWib = zeros(size(Wib));
        pmsxWfb = zeros(size(Wfb));
        pmsxWob = zeros(size(Wob));
        
        pmsxRzb = zeros(size(Rzb));
        pmsxRib = zeros(size(Rib));
        pmsxRfb = zeros(size(Rfb));
        pmsxRob = zeros(size(Rob));
        
        pmsxbzb = zeros(size(bzb));
        pmsxbib = zeros(size(bib));
        pmsxbfb = zeros(size(bfb));
        pmsxbob = zeros(size(bob));
        
        pmsxpib = zeros(size(pib));
        pmsxpfb = zeros(size(pfb));
        pmsxpob = zeros(size(pob));
        
        pmsxUb  = zeros(size(Ub));
        
        
    case 'adam'
        
        pmWzf = zeros(size(Wzf));
        pmWif = zeros(size(Wif));
        pmWff = zeros(size(Wff));
        pmWof = zeros(size(Wof));
        
        pmRzf = zeros(size(Rzf));
        pmRif = zeros(size(Rif));
        pmRff = zeros(size(Rff));
        pmRof = zeros(size(Rof));
        
        pmbzf = zeros(size(bzf));
        pmbif = zeros(size(bif));
        pmbff = zeros(size(bff));
        pmbof = zeros(size(bof));
        
        pmpif = zeros(size(pif));
        pmpff = zeros(size(pff));
        pmpof = zeros(size(pof));
        
        pmUf  = zeros(size(Uf));
        pmbu = zeros(size(bu));
        
        
        pvWzf = zeros(size(Wzf));
        pvWif = zeros(size(Wif));
        pvWff = zeros(size(Wff));
        pvWof = zeros(size(Wof));
        
        pvRzf = zeros(size(Rzf));
        pvRif = zeros(size(Rif));
        pvRff = zeros(size(Rff));
        pvRof = zeros(size(Rof));
        
        pvbzf = zeros(size(bzf));
        pvbif = zeros(size(bif));
        pvbff = zeros(size(bff));
        pvbof = zeros(size(bof));
        
        pvpif = zeros(size(pif));
        pvpff = zeros(size(pff));
        pvpof = zeros(size(pof));
        
        pvUf  = zeros(size(Uf));
        pvbu = zeros(size(bu));
        
        pmWzb = zeros(size(Wzb));
        pmWib = zeros(size(Wib));
        pmWfb = zeros(size(Wfb));
        pmWob = zeros(size(Wob));
        
        pmRzb = zeros(size(Rzb));
        pmRib = zeros(size(Rib));
        pmRfb = zeros(size(Rfb));
        pmRob = zeros(size(Rob));
        
        pmbzb = zeros(size(bzb));
        pmbib = zeros(size(bib));
        pmbfb = zeros(size(bfb));
        pmbob = zeros(size(bob));
        
        pmpib = zeros(size(pib));
        pmpfb = zeros(size(pfb));
        pmpob = zeros(size(pob));
        
        pmUb  = zeros(size(Ub));
        
        
        pvWzb = zeros(size(Wzb));
        pvWib = zeros(size(Wib));
        pvWfb = zeros(size(Wfb));
        pvWob = zeros(size(Wob));
        
        pvRzb = zeros(size(Rzb));
        pvRib = zeros(size(Rib));
        pvRfb = zeros(size(Rfb));
        pvRob = zeros(size(Rob));
        
        pvbzb = zeros(size(bzb));
        pvbib = zeros(size(bib));
        pvbfb = zeros(size(bfb));
        pvbob = zeros(size(bob));
        
        pvpib = zeros(size(pib));
        pvpfb = zeros(size(pfb));
        pvpob = zeros(size(pob));
        
        pvUb  = zeros(size(Ub));
        
end
