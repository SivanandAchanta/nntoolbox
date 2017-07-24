function [sm,hm] = fp_mmha(X,Y,p,num_heads,nout,dk)

% masked multi-head attention (used in the decoder self attention)

X = X';
Y = Y';

hm = zeros(size(Y'));
sm = zeros(size(Y,2),size(X,2),num_heads);

Qy = p.Uq*Y;
Kx = p.Uk*X;
Vx = p.Uv*X;

for i = 1:num_heads

[sm(:,:,i) hm(:,(i-1)*dk:i*dk)] = sdp_att(Qy((i-1)*dk:i*dk,:),Kx((i-1)*dk:i*dk,:),Vx((i-1)*dk:i*dk,:),dk);

end

end


function [S, H] = sdp_att(Q,K,V,dk);


B = (Q'K)/sqrt(dk);
S = get_actf('M',B); 
H = S*V';

end
