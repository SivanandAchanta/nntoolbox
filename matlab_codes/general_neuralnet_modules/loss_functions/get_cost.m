% Cost Funtion
switch cfn
    case 'nll_spk'
        me = compute_zerooneloss_spk(ym,Y);
    case 'nll'
        me = compute_zerooneloss(ym,Y);
    case 'ls'
        me = compute_nmlMSE(ym,Y);
    case 'l1_norm'
        me = compute_nmlMSE(ym,Y);
end
