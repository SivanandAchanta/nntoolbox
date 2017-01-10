
function[hc_ns,alpha_t] = compute_fw(H, lr_fw, dr_fw, st, k)

lam_t = ((dr_fw).^(k-(1:k)))';
gamma_t = (H*st);
alpha_t = gamma_t.*lam_t;
att_t = H'*alpha_t;
hc_ns = lr_fw*att_t;

end
