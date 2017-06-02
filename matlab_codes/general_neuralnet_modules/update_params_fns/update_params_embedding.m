function [Em] = update_params_embedding(Em,ix_vec,Eb,sgd_type,lr,mf,rho_hp,eps_hp,alpha,beta1,beta2,lam,num_up)


switch sgd_type   
    
    case 'sgdcm'        
        Em(ix_vec(:),:) = Em(ix_vec(:),:) - lr*Eb;
        
    case 'adadelta'
        alpha = 1e-3;
        Em(ix_vec(:),:) = Em(ix_vec(:),:) - alpha*Eb;
        
    case 'adam'
        Em(ix_vec(:),:) = Em(ix_vec(:),:) - alpha*Eb;
        
end



