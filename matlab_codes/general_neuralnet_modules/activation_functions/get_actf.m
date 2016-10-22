function [ac] = get_actf(f,pac,a_tanh,b_tanh)

switch f
    case 'N'
        ac = a_tanh*tanh(b_tanh*pac);
    case 'S'
        ac = 1./(1+(a_tanh*exp(-(b_tanh*pac))));
    case 'R' % ReLU components
        ac = max(0,pac);
    case 'Q'
        ac = min(max(0,pac),20);
    case 'E'
        ac = max(0,pac);
        ac(pac<=0) = exp(pac(pac<=0))-1;
    case 'P'
        abyb = a_p./b_p;
        ac = abyb.*(max(0,pac));
        ac_bp = pac./b_p;
        ac(pac<=0) = a_p(pac<=0).*(exp(ac_bp(pac<=0))-1);        
    case 'M' % Softmax layer
        intout = exp(pac);
        ac = bsxfun(@rdivide,intout,sum(intout,2));
    case 'L'
        ac = pac;
    otherwise
        disp('error: please enter a valid output function name (N/S/R/Q/E/P/M/L)');
        return;
end
