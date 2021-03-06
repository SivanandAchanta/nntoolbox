switch sgd_type
    case 'sgdcm'
        arch_name2 = strcat(arch_name1,'_lr',num2str(lr),'_mf',num2str(mf));
    case 'adadelta'
        arch_name2 = strcat(arch_name1,'_rho',num2str(rho_hp),'_eps',num2str(eps_hp));
    case 'adam'
        arch_name2 = strcat(arch_name1,'_alpha',num2str(alpha),'_beta1',num2str(beta1),'_beta2',num2str(beta2));
end
