
% add l1 and l2 regularization terms to the gradients
gbi = (gbi + l1*sign(Gbi) + l2*Gbi);
gba = (gba + l1*sign(Gba) + l2*Gba);
gbh = (gbh + l1*sign(Gbh) + l2*Gbh);
gbo = (gbo + l1*sign(Gbo) + l2*Gbo);
gWi = (gWi + l1*sign(GWi) + l2*GWi);
gWa = (gWa + l1*sign(GWa) + l2*GWa);
gWh = (gWh + l1*sign(GWh) + l2*GWh);
gWo = (gWo + l1*sign(GWo) + l2*GWo);

switch sgd_type
    
    case 'sgdcm'
        
        Gdbi = -lr*gbi;
        Gdba = -lr*gba;
        Gdbh = -lr*gbh;
        Gdbo = -lr*gbo;
        GdWi = -lr*gWi;
        GdWa = -lr*gWa;
        GdWh = -lr*gWh;
        GdWo = -lr*gWo;
        
        Gpdbi = Gdbi + mf*Gpdbi;
        Gpdba = Gdba + mf*Gpdba;
        Gpdbh = Gdbh + mf*Gpdbh;
        Gpdbo = Gdbo + mf*Gpdbo;
        GpdWi = GdWi + mf*GpdWi;
        GpdWa = GdWa + mf*GpdWa;
        GpdWh = GdWh + mf*GpdWh;
        GpdWo = GdWo + mf*GpdWo;
        
        % Update weights
        Gbi = Gbi + Gpdbi;
        Gba = Gba + Gpdba;
        Gbh = Gbh + Gpdbh;
        Gbo = Gbo + Gpdbo;
        GWi = GWi + GpdWi;
        GWa = GWa + GpdWa;
        GWh = GWh + GpdWh;
        GWo = GWo + GpdWo;
        
    case 'adadelta'
        
        % adadelta equations
        Gpmsgbi = rho_hp*(Gpmsgbi) + (1-rho_hp)*(gbi.^2);
        Gpmsgba = rho_hp*(Gpmsgba) + (1-rho_hp)*(gba.^2);
        Gpmsgbh = rho_hp*(Gpmsgbh) + (1-rho_hp)*(gbh.^2);
        Gpmsgbo = rho_hp*(Gpmsgbo) + (1-rho_hp)*(gbo.^2);
        GpmsgWi = rho_hp*(GpmsgWi) + (1-rho_hp)*(gWi.^2);
        GpmsgWa = rho_hp*(GpmsgWa) + (1-rho_hp)*(gWa.^2);
        GpmsgWh = rho_hp*(GpmsgWh) + (1-rho_hp)*(gWh.^2);
        GpmsgWo = rho_hp*(GpmsgWo) + (1-rho_hp)*(gWo.^2);
        
        %
        rmsgbi = sqrt(Gpmsgbi + eps_hp);  rmsxbi = sqrt(Gpmsxbi + eps_hp);
        rmsgba = sqrt(Gpmsgba + eps_hp);  rmsxba = sqrt(Gpmsxba + eps_hp);
        rmsgbh = sqrt(Gpmsgbh + eps_hp);  rmsxbh = sqrt(Gpmsxbh + eps_hp);
        rmsgbo = sqrt(Gpmsgbo + eps_hp);  rmsxbo = sqrt(Gpmsxbo + eps_hp);
        rmsgWi = sqrt(GpmsgWi + eps_hp); rmsxWi = sqrt(GpmsxWi + eps_hp);
        rmsgWa = sqrt(GpmsgWa + eps_hp); rmsxWa = sqrt(GpmsxWa + eps_hp);
        rmsgWh = sqrt(GpmsgWh + eps_hp); rmsxWh = sqrt(GpmsxWh + eps_hp);
        rmsgWo = sqrt(GpmsgWo + eps_hp); rmsxWo = sqrt(GpmsxWo + eps_hp);
        
        Gdbi = -(rmsxbi./rmsgbi).*gbi;
        Gdba = -(rmsxba./rmsgba).*gba;
        Gdbh = -(rmsxbh./rmsgbh).*gbh;
        Gdbo = -(rmsxbo./rmsgbo).*gbo;
        GdWi = -(rmsxWi./rmsgWi).*gWi;
        GdWa = -(rmsxWa./rmsgWa).*gWa;
        GdWh = -(rmsxWh./rmsgWh).*gWh;
        GdWo = -(rmsxWo./rmsgWo).*gWo;
        
        
        %
        Gpmsxbi = rho_hp*(Gpmsxbi) + (1-rho_hp)*(Gdbi.^2);
        Gpmsxba = rho_hp*(Gpmsxba) + (1-rho_hp)*(Gdba.^2);
        Gpmsxbh = rho_hp*(Gpmsxbh) + (1-rho_hp)*(Gdbh.^2);
        Gpmsxbo = rho_hp*(Gpmsxbo) + (1-rho_hp)*(Gdbo.^2);
        GpmsxWi = rho_hp*(GpmsxWi) + (1-rho_hp)*(GdWi.^2);
        GpmsxWa = rho_hp*(GpmsxWa) + (1-rho_hp)*(GdWa.^2);
        GpmsxWh = rho_hp*(GpmsxWh) + (1-rho_hp)*(GdWh.^2);
        GpmsxWo = rho_hp*(GpmsxWo) + (1-rho_hp)*(GdWo.^2);
        
        Gpdbi = Gdbi + mf*Gpdbi;
        Gpdba = Gdba + mf*Gpdba;
        Gpdbh = Gdbh + mf*Gpdbh;
        Gpdbo = Gdbo + mf*Gpdbo;
        GpdWi = GdWi + mf*GpdWi;
        GpdWa = GdWa + mf*GpdWa;
        GpdWh = GdWh + mf*GpdWh;
        GpdWo = GdWo + mf*GpdWo;
        
        % Update weights
        Gbi = Gbi + Gpdbi;
        Gba = Gba + Gpdba;
        Gbh = Gbh + Gpdbh;
        Gbo = Gbo + Gpdbo;
        GWi = GWi + GpdWi;
        GWa = GWa + GpdWa;
        GWh = GWh + GpdWh;
        GWo = GWo + GpdWo;
        
    case 'adam'
        
        % Adam equations
        beta1_t = beta1*(lam^(num_up-1));
        
        % biased moments
        Gpmbi = beta1_t*Gpmbi + (1 - beta1_t)*gbi;
        Gpmba = beta1_t*Gpmba + (1 - beta1_t)*gba;
        Gpmbh = beta1_t*Gpmbh + (1 - beta1_t)*gbh;
        Gpmbo = beta1_t*Gpmbo + (1 - beta1_t)*gbo;
        GpmWi = beta1_t*GpmWi + (1 - beta1_t)*gWi;
        GpmWa = beta1_t*GpmWa + (1 - beta1_t)*gWa;
        GpmWh = beta1_t*GpmWh + (1 - beta1_t)*gWh;
        GpmWo = beta1_t*GpmWo + (1 - beta1_t)*gWo;
        
        Gpvbi = beta2*Gpvbi + (1 - beta2)*(gbi.^2);
        Gpvba = beta2*Gpvba + (1 - beta2)*(gba.^2);
        Gpvbh = beta2*Gpvbh + (1 - beta2)*(gbh.^2);
        Gpvbo = beta2*Gpvbo + (1 - beta2)*(gbo.^2);
        GpvWi = beta2*GpvWi + (1 - beta2)*(gWi.^2);
        GpvWa = beta2*GpvWa + (1 - beta2)*(gWa.^2);
        GpvWh = beta2*GpvWh + (1 - beta2)*(gWh.^2);
        GpvWo = beta2*GpvWo + (1 - beta2)*(gWo.^2);
        
        % bias correction
        mbit_cap = Gpmbi./(1 - beta1^num_up);
        mbat_cap = Gpmba./(1 - beta1^num_up);
        mbht_cap = Gpmbh./(1 - beta1^num_up);
        mbot_cap = Gpmbo./(1 - beta1^num_up);
        mWit_cap = GpmWi./(1 - beta1^num_up);
        mWat_cap = GpmWa./(1 - beta1^num_up);
        mWht_cap = GpmWh./(1 - beta1^num_up);
        mWot_cap = GpmWo./(1 - beta1^num_up);
        
        vbit_cap = Gpvbi./(1 - beta2^num_up);
        vbat_cap = Gpvba./(1 - beta2^num_up);
        vbht_cap = Gpvbh./(1 - beta2^num_up);
        vbot_cap = Gpvbo./(1 - beta2^num_up);
        vWit_cap = GpvWi./(1 - beta2^num_up);
        vWat_cap = GpvWa./(1 - beta2^num_up);
        vWht_cap = GpvWh./(1 - beta2^num_up);
        vWot_cap = GpvWo./(1 - beta2^num_up);
        
        % Update Params
        Gbi = Gbi - alpha_adam*mbit_cap./(sqrt(vbit_cap)+eps_hp);
        Gba = Gba - alpha_adam*mbat_cap./(sqrt(vbat_cap)+eps_hp);
        Gbh = Gbh - alpha_adam*mbht_cap./(sqrt(vbht_cap)+eps_hp);
        Gbo = Gbo - alpha_adam*mbot_cap./(sqrt(vbot_cap)+eps_hp);
        GWi = GWi - alpha_adam*mWit_cap./(sqrt(vWit_cap)+eps_hp);
        GWa = GWa - alpha_adam*mWat_cap./(sqrt(vWat_cap)+eps_hp);
        GWh = GWh - alpha_adam*mWht_cap./(sqrt(vWht_cap)+eps_hp);
        GWo = GWo - alpha_adam*mWot_cap./(sqrt(vWot_cap)+eps_hp);
        
end



