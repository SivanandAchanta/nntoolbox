
% weight initialization
wt_init

% get full arch_name
get_fullarchname

% Step 5 : Training
if gpu_flag
    train_gpu
else
    train_cpu
end
