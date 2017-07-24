% Encoder Layer - 1
[sm_mha_1_1, hm_mha_1_1] = fp_mha(X,X,p_mha_1,num_heads,d_model,dk); % multi-head attention layer (self-attention)
hm_mha_1_2 = hm_mha_1_1 + X; % residual layer
[hm_mha_1_3,mu_mha_1_3,sig_mha_1_3] = fp_ln(hm_mha_1_2,p_ln_1_1); % layer normalization

hm_ff_1_1 = fp_cpu_ll(hm_mha_1_3,p_ff_1_1,'R'); % linar T/F 1 followed by ReLU
hm_ff_1_2 = fp_cpu_ll(hm_ff_1_1,p_ff_1_2,'L'); % linear T/F 2
hm_ff_1_3 = hm_ff_1_2 + hm_mha_1_3; % residual layer
[hm_ff_1_4,mu_ff_1_4,sig_ff_1_4] = fp_ln(hm_ff_1_3,p_ln_1_2); % layer normalization


% Decoder Layer - 1
[sm_mha_1_1_dec, hm_mha_1_1_dec] = fp_mha(Y,Y,p_mha_1_dec,num_heads,d_model,dk); % multi-head attention layer (self-attention)
hm_mha_1_2_dec = hm_mha_1_1_dec + Y; % residual layer
[hm_mha_1_3_dec,mu_mha_1_3_dec,sig_mha_1_3_dec] = fp_ln(hm_mha_1_2_dec,p_ln_1_1_dec); % layer normalization

[sm_mha_1_4_dec, hm_mha_1_4_dec] = fp_mha(X,hm_mha_1_3_dec,p_mha_2_dec,num_heads,d_model,dk); % multi-head attention layer (enc-dec attention)
hm_mha_1_5_dec = hm_mha_1_4_dec + hm_mha_1_3_dec; % residual layer
[hm_mha_1_6_dec,mu_mha_1_6_dec,sig_mha_1_6_dec] = fp_ln(hm_mha_1_5_dec,p_ln_1_2_dec); % layer normalization

hm_ff_1_7_dec = fp_cpu_ll(hm_mha_1_6_dec,p_ff_1_1_dec,'R'); % linar T/F 1 followed by ReLU
hm_ff_1_8_dec = fp_cpu_ll(hm_ff_1_7_dec,p_ff_1_2_dec,'L'); % linear T/F 2
hm_ff_1_9_dec = hm_ff_1_8_dec + hm_mha_1_6_dec; % residual layer
[hm_ff_1_10_dec,mu_ff_1_10_dec,sig_ff_1_10_dec] = fp_ln(hm_ff_1_9_dec,p_ln_1_3_dec); % layer normalization

% Output Layer
hm_ff_1_11_dec = fp_cpu_ll(hm_ff_1_10_dec,p_ff_1_3_dec,f(end)); % o/p layer



