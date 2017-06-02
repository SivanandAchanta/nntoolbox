function [Gp] = wt_init_ll_ob(p,gpu_flag,nout,nin,wtdir,wtinit_meth,sgd_type)

p.bu = 0.1*randn(nout,1);
disp('size of weight matrices');
size(p.bu)


switch sgd_type
    case 'sgdcm'
        if gpu_flag
            Gp.bu = gpuArray(p.bu);
            Gp.pdbu = gpuArray(zeros(size(p.bu)));
        else
            Gp.bu = p.bu;
            Gp.pdbu = zeros(size(p.bu));
        end
        
    case 'adadelta'
        if gpu_flag
            Gp.bu = gpuArray(p.bu);
            Gp.pdbu = gpuArray(zeros(size(p.bu)));
            Gp.pmsgbu = gpuArray(zeros(size(p.bu)));
            Gp.pmxgbu = gpuArray(zeros(size(p.bu)));
        else
            Gp.bu = p.bu;
            Gp.pdbu = zeros(size(p.bu));
            Gp.pmsgbu = zeros(size(p.bu));
            Gp.pmxgbu = zeros(size(p.bu));
        end
        
    case 'adam'
        if gpu_flag
            Gp.bu = gpuArray(p.bu);
            Gp.pmbu = gpuArray(zeros(size(p.bu)));
            Gp.pvbu = gpuArray(zeros(size(p.bu)));
        else
            Gp.bu = p.bu;
            Gp.pmbu = zeros(size(p.bu));
            Gp.pvbu = zeros(size(p.bu));
        end
        
end
