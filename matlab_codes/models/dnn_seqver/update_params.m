
switch sgd_type
    
    case 'sgdcm'
        
        [GW, GpdW] = sgdcm(lr, mf, gW, GpdW, GW);
        [Gb, Gpdb] = sgdcm(lr, mf, gb, Gpdb, Gb);
        
    case 'adadelta'
        
        [GW, GpmsgW, GpmsxW, GpdW] = adadelta(rho_hp, eps_hp, mf, gW, GpmsgW, GpmsxW, GpdW, GW);
        [Gb, Gpmsgb, Gpmsxb, Gpdb] = adadelta(rho_hp, eps_hp, mf, gb, Gpmsgb, Gpmsxb, Gpdb, Gb);
        
    case 'adam'
        
        [GW,GpmW,GpvW]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gW,GpmW,GpvW,GW);
        [Gb,Gpmb,Gpvb]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gb,Gpmb,Gpvb,Gb);
        
end



