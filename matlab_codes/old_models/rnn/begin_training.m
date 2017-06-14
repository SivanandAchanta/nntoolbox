% weight initialization
wt_init

% get full arch_name
get_fullarchname

% train the model
if gpu_flag
    trainrnn_gpu
else
    trainrnn_cpu
end
