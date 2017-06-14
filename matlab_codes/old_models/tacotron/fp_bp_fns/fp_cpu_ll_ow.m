function [ym] = fp_cpu_ll_ow(X,p,f)

U = p.U;

ac = U*X';
ac = ac';

ym = get_actf(f,ac);