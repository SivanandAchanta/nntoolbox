function [der_f,varargout] = get_derf(nl,f,hcm,sl,varargin)

% set params of nonlinearity
a_tanh = 1.7159;
b_tanh = 2/3;
bby2a = (b_tanh/(2*a_tanh));

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
        apelu = varargin{1};
        bpelu = varargin{2};
        
        abyb = apelu./bpelu;
        der_f = bsxfun(@times,ones(sl,nl),abyb').*(hcm > 0);
        t1 = bsxfun(@plus,hcm,apelu');
        t2 = bsxfun(@rdivide,t1,bpelu');
        der_f(hcm<0) = t2(hcm<0);
        
        dhcm_ap = bsxfun(@rdivide,hcm,apelu');
        varargout{1} = dhcm_ap;
        
        t3 = bsxfun(@rdivide,-hcm,bpelu');
        t1bya = bsxfun(@rdivide,t1,apelu');
        t1bya(t1bya<=0) = 1e-8;
        xbyb = log(t1bya);
        thcm4 = -xbyb.*t2;
        
        dhcm_bp = t3;
        dhcm_bp(hcm<0) = thcm4(hcm<0);
        varargout{2} = dhcm_bp;
        
    case 'M' % Softmax layer
        der_f           = (hcm.*(1 - hcm));
    case 'L'
        der_f           = ones(sl,nl);
    otherwise
        disp('please enter a valid output function name (N/S/R/M/L)');
        return;
end
