
function[bln] = bp_ln(pac,sig,eyem,dhi,gn)

t1 = eyem;
t2 = -dhi*(pac*pac'); 

bln = (t1 + t2)/sig;

bln = bsxfun(@times,bln,gn);

end