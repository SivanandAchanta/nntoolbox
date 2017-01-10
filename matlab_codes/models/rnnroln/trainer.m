
% Train RNN
if gpu_flag
    if gt_flag
        trainrnn_gpu_gt
    else
        trainrnn_gpu_pt
    end
else
    if gt_flag
        trainrnn_cpu_gt
    else
        trainrnn_cpu_pt
    end
end
