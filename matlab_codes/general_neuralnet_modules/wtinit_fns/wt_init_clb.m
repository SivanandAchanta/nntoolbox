function [Gp] = wt_init_clb(p,gpu_flag,nin,so,K,C,wtdir,wtinit_meth,sgd_type)

% k - filters with width (1 - k)
% C - number of filters at each width

switch K
    case 1
        p.U1 = so*randn(1,nin*C);
    case 2
        p.U1 = so*randn(1,nin*C);
        p.U2 = so*randn(2,nin*C);
    case 3
        k = 1; p.U1 = so*randn(k,nin*C);
        k = k + 1; p.U2 = so*randn(k,nin*C);
        k = k + 1; p.U3 = so*randn(k,nin*C);
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
                    Gp.U1 = p.U1; Gp.pmU1 = zeros(size(p.U1)); Gp.pvU1 = zeros(size(p.U1));
                    Gp.U2 = p.U2; Gp.pmU2 = zeros(size(p.U2)); Gp.pvU2 = zeros(size(p.U2));
                    Gp.U3 = p.U3; Gp.pmU3 = zeros(size(p.U3)); Gp.pvU3 = zeros(size(p.U3));
                end                
        end
        
    case 4
        k = 1; p.U1 = so*randn(k,nin*C);
        k = k + 1; p.U2 = so*randn(k,nin*C);
        k = k + 1; p.U3 = so*randn(k,nin*C);
        k = k + 1; p.U4 = so*randn(k,nin*C);
    case 5
        k = 1; p.U1 = so*randn(k,nin*C);
        k = k + 1; p.U2 = so*randn(k,nin*C);
        k = k + 1; p.U3 = so*randn(k,nin*C);
        k = k + 1; p.U4 = so*randn(k,nin*C);
        k = k + 1; p.U5 = so*randn(k,nin*C);
    case 6
        k = 1; p.U1 = so*randn(k,nin*C);
        k = k + 1; p.U2 = so*randn(k,nin*C);
        k = k + 1; p.U3 = so*randn(k,nin*C);
        k = k + 1; p.U4 = so*randn(k,nin*C);
        k = k + 1; p.U5 = so*randn(k,nin*C);
        k = k + 1; p.U6 = so*randn(k,nin*C);
    case 7
        k = 1; p.U1 = so*randn(k,nin*C);
        k = k + 1; p.U2 = so*randn(k,nin*C);
        k = k + 1; p.U3 = so*randn(k,nin*C);
        k = k + 1; p.U4 = so*randn(k,nin*C);
        k = k + 1; p.U5 = so*randn(k,nin*C);
        k = k + 1; p.U6 = so*randn(k,nin*C);
        k = k + 1; p.U7 = so*randn(k,nin*C);
    case 8
        k = 1; p.U1 = so*randn(k,nin*C);
        k = k + 1; p.U2 = so*randn(k,nin*C);
        k = k + 1; p.U3 = so*randn(k,nin*C);
        k = k + 1; p.U4 = so*randn(k,nin*C);
        k = k + 1; p.U5 = so*randn(k,nin*C);
        k = k + 1; p.U6 = so*randn(k,nin*C);
        k = k + 1; p.U7 = so*randn(k,nin*C);
        k = k + 1; p.U8 = so*randn(k,nin*C);
        
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
                    Gp.U1 = p.U1; Gp.pmU1 = zeros(size(p.U1)); Gp.pvU1 = zeros(size(p.U1));
                    Gp.U2 = p.U2; Gp.pmU2 = zeros(size(p.U2)); Gp.pvU2 = zeros(size(p.U2));
                    Gp.U3 = p.U3; Gp.pmU3 = zeros(size(p.U3)); Gp.pvU3 = zeros(size(p.U3));
                    Gp.U4 = p.U4; Gp.pmU4 = zeros(size(p.U4)); Gp.pvU4 = zeros(size(p.U4));
                    Gp.U5 = p.U5; Gp.pmU5 = zeros(size(p.U5)); Gp.pvU5 = zeros(size(p.U5));
                    Gp.U6 = p.U6; Gp.pmU6 = zeros(size(p.U6)); Gp.pvU6 = zeros(size(p.U6));
                    Gp.U7 = p.U7; Gp.pmU7 = zeros(size(p.U7)); Gp.pvU7 = zeros(size(p.U7));
                    Gp.U8 = p.U8; Gp.pmU8 = zeros(size(p.U8)); Gp.pvU8 = zeros(size(p.U8));
                end
        end
        
        
end


