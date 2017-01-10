function [pac, bac, ac, mu, sig] = layer_norm(ac, gn, bn)

mu = mean(ac);
sig = std(ac,1);

pac = ac;
bac = (ac - mu)/sig;
ac = gn.*bac + bn;

end