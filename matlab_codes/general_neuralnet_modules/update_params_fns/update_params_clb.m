function [p] = update_params_clb(p,K_conv,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up)
switch sgd_type
    
    case 'sgdcm'
        
        [p.U, p.pdU] = sgdcm(lr, mf, p.gU, p.pdU, p.U);
        
    case 'adadelta'
        
        [p.U, p.pmsgU, p.pmsxU, p.pdU] = adadelta(rho_hp, eps_hp, mf, p.gU, p.pmsgU, p.pmsxU, p.pdU, p.U);
        
    case 'adam'
        if K_conv == 3
            [p.U1,p.pmU1,p.pvU1]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU1,p.pmU1,p.pvU1,p.U1);
            [p.U2,p.pmU2,p.pvU2]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU2,p.pmU2,p.pvU2,p.U2);
            [p.U3,p.pmU3,p.pvU3]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU3,p.pmU3,p.pvU3,p.U3);
        elseif K_conv == 8
            [p.U1,p.pmU1,p.pvU1]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU1,p.pmU1,p.pvU1,p.U1);
            [p.U2,p.pmU2,p.pvU2]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU2,p.pmU2,p.pvU2,p.U2);
            [p.U3,p.pmU3,p.pvU3]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU3,p.pmU3,p.pvU3,p.U3);
            [p.U4,p.pmU4,p.pvU4]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU4,p.pmU4,p.pvU4,p.U4);
            [p.U5,p.pmU5,p.pvU5]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU5,p.pmU5,p.pvU5,p.U5);
            [p.U6,p.pmU6,p.pvU6]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU6,p.pmU6,p.pvU6,p.U6);
            [p.U7,p.pmU7,p.pvU7]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU7,p.pmU7,p.pvU7,p.U7);
            [p.U8,p.pmU8,p.pvU8]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,p.gU8,p.pmU8,p.pvU8,p.U8);
        end
        
end



