
% Weight initialization hyperparameters
switch wtinit_meth
    case 'rg'
        si = 0.2;
        sa = 0.2;
        sh = 0.2;
        so = 0.1;
end

wthyper_str = strcat('si_',num2str(si),'sa_',num2str(sa),'sh_',num2str(sh),'so_',num2str(so));
