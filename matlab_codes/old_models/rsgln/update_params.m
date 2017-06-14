
switch sgd_type
    
    case 'sgdcm'
        
        [GWi, GpdWi] = sgdcm(lr, mf, gWi, GpdWi, GWi);
        [GWfr, GpdWfr] = sgdcm(lr, mf, gWfr, GpdWfr, GWfr);
        [Gbh, Gpdbh] = sgdcm(lr, mf, gbh, Gpdbh, Gbh);
        [GU, GpdU] = sgdcm(lr, mf, gU, GpdU, GU);
        [GWy, GpdWy] = sgdcm(lr, mf, gWy, GpdWy, GWy);
        [Gbo, Gpdbo] = sgdcm(lr, mf, gbo, Gpdbo, Gbo);
        [Ggn, Gpdgn] = sgdcm(lr, mf, ggn, Gpdgn, Ggn);
        [Gbn, Gpdbn] = sgdcm(lr, mf, gbn, Gpdbn, Gbn);
    case 'adadelta'
        
        [GWi, GpmsgWi, GpmsxWi, GpdWi] = adadelta(rho, eps_hp, mf, gWi, GpmsgWi, GpmsxWi, GpdWi, GWi);
        [GWfr, GpmsgWfr, GpmsxWfr, GpdWfr] = adadelta(rho, eps_hp, mf, gWfr, GpmsgWfr, GpmsxWfr, GpdWfr, GWfr);
        [Gbh, Gpmsgbh, Gpmsxbh, Gpdbh] = adadelta(rho, eps_hp, mf, gbh, Gpmsgbh, Gpmsxbh, Gpdbh, Gbh);
        [GU, GpmsgU, GpmsxU, GpdU] = adadelta(rho, eps_hp, mf, gU, GpmsgU, GpmsxU, GpdU, GU);
        [GWy, GpmsgWy, GpmsxWy, GpdWy] = adadelta(rho, eps_hp, mf, gWy, GpmsgWy, GpmsxWy, GpdWy, GWy);
        [Gbo, Gpmsgbo, Gpmsxbo, Gpdbo] = adadelta(rho, eps_hp, mf, gbo, Gpmsgbo, Gpmsxbo, Gpdbo, Gbo);
        [Ggn, Gpmsggn, Gpmsxgn, Gpdgn] = adadelta(rho, eps_hp, mf, ggn, Gpmsggn, Gpmsxgn, Gpdgn, Ggn);
        [Gbn, Gpmsgbn, Gpmsxbn, Gpdbn] = adadelta(rho, eps_hp, mf, gbn, Gpmsgbn, Gpmsxbn, Gpdbn, Gbn);
    case 'adam'
        
        [GWi,GpmWi,GpvWi]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gWi,GpmWi,GpvWi,GWi);
        [GWfr,GpmWfr,GpvWfr]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gWfr,GpmWfr,GpvWfr,GWfr);
        [Gbh,Gpmbh,Gpvbh]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gbh,Gpmbh,Gpvbh,Gbh);
        [GU,GpmU,GpvU]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gU,GpmU,GpvU,GU);
        [GWy,GpmWy,GpvWy]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gWy,GpmWy,GpvWy,GWy);
        [Gbo,Gpmbo,Gpvbo]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gbo,Gpmbo,Gpvbo,Gbo);
        
        [Ggn,Gpmgn,Gpvgn]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,ggn,Gpmgn,Gpvgn,Ggn);
        [Gbn,Gpmbn,Gpvbn]      = adam(alpha,beta1,beta2,eps_hp,lam,num_up,gbn,Gpmbn,Gpvbn,Gbn);
end



