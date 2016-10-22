function [X,Y,bs] = get_XY_seqver(data, targets, clv, rp, i, gpu_flag)

X = data(clv(rp(i)):clv(rp(i)+1)-1,:);
Y = targets(clv(rp(i)):clv(rp(i)+1)-1,:);
bs = size(X,1);

X = double(X);
Y = double(Y);

if gpu_flag
    X = gpuArray(X);
    Y = gpuArray(Y);
end
    

end