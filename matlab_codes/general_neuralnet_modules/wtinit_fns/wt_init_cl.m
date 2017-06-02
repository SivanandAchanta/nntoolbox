function [Gp] = wt_init_cl(p,gpu_flag,nin,so,K,C,wtdir,wtinit_meth,sgd_type)

% k - filters with width (1 - k)
% C - number of filters at each width

p.U = so*randn(K,nin*C);

switch sgd_type
    case 'sgdcm'
        if gpu_flag
            Gp.U = gpuArray(p.U); Gp.pdU = gpuArray(zeros(size(p.U)));
        else
            Gp.U = p.U; Gp.pdU = zeros(size(p.U));
        end
    case 'adadelta'
        if gpu_flag
            Gp.U = gpuArray(p.U); Gp.pdU = gpuArray(zeros(size(p.U))); Gp.pmsgU = gpuArray(zeros(size(p.U))); Gp.pmxgU = gpuArray(zeros(size(p.U)));
        else
            Gp.U = p.U; Gp.pdU = zeros(size(p.U)); Gp.pmsgU = zeros(size(p.U)); Gp.pmxgU = zeros(size(p.U));
        end
    case 'adam'
        if gpu_flag
            Gp.U = gpuArray(p.U); Gp.pmU = gpuArray(zeros(size(p.U))); Gp.pvU = gpuArray(zeros(size(p.U)));
        else
            Gp.U = p.U; Gp.pmU = zeros(size(p.U)); Gp.pvU = zeros(size(p.U));
        end
end


end


