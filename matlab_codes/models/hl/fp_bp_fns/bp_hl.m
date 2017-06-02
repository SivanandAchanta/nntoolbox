function [gW,gb,gWt,gbt,Eb] = bp_hl(nout,sl,E,p,htm,tm,X)

W = p.W;
Wt = p.Wt;

derhtm = get_derf(nout,'R',htm,sl);
dertm = tm.*(1-tm);

dhtm = derhtm.*(E.*tm);
dtm = dertm.*(E.*(htm - X));

gW = (dhtm'*X);
gb = sum(dhtm,1)';

gWt = (dtm'*X);
gbt = sum(dtm,1)';

Eb = (1-tm).*E + dhtm*W + dtm*Wt; 

end