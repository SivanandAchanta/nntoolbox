function [tot_err] = compute_error(batch_data,Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,nl,sl,ff,U,bu,numbats,cfn)

tot_err = 0;

for li  = 1:numbats
    
    X   = get_X(batch_data,li);
    
    % Forward Pass
    [~,~,~,~,~,~,~,ym] = fp_lstmsg_test(Wz,Rz,bz,Wi,Ri,pi,bi,Wf,Rf,pf,bf,Wo,Ro,po,bo,nl,sl,ff,U,bu);
    
    % Cost Funtion
    switch cfn
        case 'ls'
            J   = mean(sum(power((X - ym),2),2));
        case  'nll'
            J   = mean(sum((-X.*log(ym)),2));
            
    end
    
    tot_err     = tot_err + J/numbats;
    
end