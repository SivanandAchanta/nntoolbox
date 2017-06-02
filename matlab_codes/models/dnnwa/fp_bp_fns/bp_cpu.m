% bp
dely = -(Y-ym)/sl;
gWo = dely'*sm;
gbo = sum(dely,1)';

[der_f] = get_derf(nl(3),f(2),sm,sl);
dels = (der_f.*(dely*GWo)); % 1 x d
gWh = dels'*c'; % d x d 
gbh = sum(dels,1)'; % d x 1 

delc = dels*GWh; % 1 x d
dela = delc*hcm'; % 1 x N
der_f_alpha = (diag(alpha) - alpha*alpha'); % sle x sle
delb = dela*der_f_alpha; % 1 x sle
der_f = get_derf(nl(2),'N',ah,sl); % sle x 1
delbb = der_f'.*delb; % 1 x sle
gWa = delbb*hcm; % 1xd
gba = sum(delbb,2)';

iemata = delbb'*GWa;
iematc = alpha*delc;
[der_f] = get_derf(nl(2),f(1),hcm,sl_enc);
deli = der_f.*(iemata+iematc);
gWi = deli'*X;
gbi = sum(deli,1)';