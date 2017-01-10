
% Train RNN
if gpu_flag
    if gt_flag
        trainrsg_gpu_gt
    else
        trainrsg_gpu_pt
    end
else
    if gt_flag
        trainrsg_cpu_gt
    else
        trainrsg_cpu_pt
    end
end
