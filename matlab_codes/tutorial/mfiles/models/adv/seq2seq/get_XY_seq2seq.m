function [X, Yf, Yp, sl_enc, sl_dec] = get_XY_tacotron(train_batchdata, train_batchtargets, train_clv_s, train_clv_t, rp, i)

X = train_batchdata(train_clv_s(rp(i)):train_clv_s(rp(i)+1)-1,:);
%X = randperm(5,4)';
sl_enc = size(X,1);

%Xe = Em(X(:),:);

Y = train_batchtargets(train_clv_t(rp(i)):train_clv_t(rp(i)+1)-1,:);
%Y = randn(10,3);
sl_dec = size(Y,1);

% Put "r" frames together
if mod(sl_dec,2) == 0
    Ye = Y(2:2:end,:);
    Yo = Y(1:2:end,:);
else
    Y = Y(1:end-1,:);
    Ye = Y(2:2:end,:);
    Yo = Y(1:2:end,:);
end

Yf = [Ye Yo];
sl_dec = size(Yf,1);

Yp = [zeros(1,size(Ye,2));Ye(1:end-1,:)];

end
