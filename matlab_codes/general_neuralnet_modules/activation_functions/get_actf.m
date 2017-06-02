function [ac] = get_actf(f,pac,varargin)

% pac - N x d
% ac - N x d

% set params of nonlinearity
a_tanh = 1.7159;
b_tanh = 2/3;


switch f
    case 'N'
        ac = a_tanh*tanh(b_tanh*pac);
    case 'S'
        ac = 1./(1+(a_tanh*exp(-(b_tanh*pac))));
    case 'sigm'
        ac = 1./(1+(exp(-(pac))));
    case 'R' % ReLU components
        ac = max(0,pac);
    case 'Q'
        ac = min(max(0,pac),20);
    case 'E'
        ac = max(0,pac);
        ac(pac<=0) = exp(pac(pac<=0))-1;
    case 'P'
        apelu = varargin{1};
        bpelu = varargin{2};
        
        abyb = apelu./bpelu;
        ac = abyb.*(max(0,pac));
        ac_bp = pac./bpelu;
        ac(pac<=0) = apelu(pac<=0).*(exp(ac_bp(pac<=0))-1);
    case 'M' % Softmax layer
        intout = exp(pac);
        ac = bsxfun(@rdivide,intout,sum(intout,2));
    case 'L'
        ac = pac;
    otherwise
        disp('error: please enter a valid output function name (N/S/R/Q/E/P/M/L)');
        return;
end
