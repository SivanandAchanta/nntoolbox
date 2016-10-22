function [der_f] = get_derf(nl,f,hcm,sl,a_tanh,b_tanh,bby2a)

switch f
    case 'N'
        der_f           = 2*bby2a*((a_tanh - hcm).*(a_tanh + hcm));
    case 'S'
        der_f           = b_tanh*(hcm.*(1 - hcm));
    case 'R' % added on 28/11/14
        der_f           = ones(sl,nl).*(hcm > 0);
    case 'E'
        der_f           = ones(sl,nl).*(hcm > 0);
        der_f(hcm<=0)   = hcm(hcm<=0) + 1;
    case 'Q'
        der_f           = ones(sl,nl).*(hcm > 0).*(hcm < 20);        
    case 'P'
        der_f           = ones(sl,nl).*(hcm > 0);
        der_f(hcm<=0)   = hcm(hcm<=0) + 1;
    case 'M' % Softmax layer
        der_f           = (hcm.*(1 - hcm));
    case 'L'
        der_f           = ones(sl,nl);
    otherwise
        disp('please enter a valid output function name (N/S/R/M/L)');
        return;
end
