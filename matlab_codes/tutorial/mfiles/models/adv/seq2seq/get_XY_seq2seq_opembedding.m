function [X, Ye, Yp, sl_enc, sl_dec] = get_XY_seq2seq_opembedding(train_batchdata, train_batchtargets, train_clv_s, train_clv_t, rp, i, Em)

X = train_batchdata(train_clv_s(rp(i)):train_clv_s(rp(i)+1)-1,:);
sl_enc = size(X,1);

Y = train_batchtargets(train_clv_t(rp(i)):train_clv_t(rp(i)+1)-1,:);
sl_dec = size(Y,1);

Ye = Em(Y(:),:);
Yp = [zeros(1,size(Ye,2));Ye(1:end-1,:)];

end
