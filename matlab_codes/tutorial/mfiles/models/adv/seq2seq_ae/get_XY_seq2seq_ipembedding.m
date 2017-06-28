function [Xe, Y, Yp, sl_enc, sl_dec] = get_XY_seq2seq_ipembedding(train_batchdata, train_batchtargets, train_clv_s, train_clv_t, rp, i, Em)

X = train_batchdata(train_clv_s(rp(i)):train_clv_s(rp(i)+1)-1,:);
sl_enc = size(X,1);
Xe = Em(X(:),:);

Y = Xe;
sl_dec = sl_enc;

%Y = train_batchtargets(train_clv_t(rp(i)):train_clv_t(rp(i)+1)-1,:);
%sl_dec = size(Y,1);

% Put "r" frames together
%if mod(sl_dec,2) == 0
%    Ye = Y(2:2:end,:);
%    Yo = Y(1:2:end,:);
%else
%    Y = Y(1:end-1,:);
%    Ye = Y(2:2:end,:);
%    Yo = Y(1:2:end,:);
%end

%Yf = [Ye Yo];
%sl_dec = size(Yf,1);

Yp = [zeros(1,size(Y,2));Y(1:end-1,:)];

end
