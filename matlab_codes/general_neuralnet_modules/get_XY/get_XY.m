function [X, Y] = get_XY(data, targets, rp, i, gpu_flag)

X = data(:,:,rp(i));
Y = targets(:,:,rp(i));

X = double(X);
Y = double(Y);

if gpu_flag
   X = gpuArray(X);
   Y = gpuArray(Y);
end

end

