% Error at Output Layer
switch cfn
    case  'nll_spk'
        E   = -(Y - ym)/sl;
    case  'nll'
        E   = -(Y - ym)/sl;
   case 'ls'
        E   = -(Y - ym)/sl;
    case 'l1_norm'
        E   = -sign(Y - ym)/sl;
end

