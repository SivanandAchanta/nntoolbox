

% Set the input output dimensions and the normalization flags
invec = [1:347];
mvnivec = [303:339 343:347];

feat_name = 'cmp';
if strcmp(feat_name,'mgc')
    outvec = [1:150];
elseif strcmp(feat_name,'f0')
    outvec = [232:235];
elseif strcmp(feat_name,'bap')
    outvec = [151:228];
elseif strcmp(feat_name,'cmp')
    outvec = [1:235];
end

in_nml_meth = 'mvni'
out_nml_meth = 'mvno'


% set synthetic data dimensions for gradient checking
if gradCheckFlag
    invec = [1:5];
    outvec = [1:3];
    din = length(invec);
    dout = length(outvec);
    in_nml_meth = ''
    out_nml_meth = ''
end


if strcmp(in_nml_meth,'mvni')
   intmvnf = 1;
   intmmnf = 0;
elseif strcmp(in_nml_meth,'mmni')
   intmvnf = 0;
   intmmnf = 1;
else
   intmvnf = 0;
   intmmnf = 0;
end

if strcmp(out_nml_meth,'mvno')
   outtmvnf = 1;
   outtmmnf = 0;
elseif strcmp(out_nml_meth,'mmno')
   outtmvnf = 0;
   outtmmnf = 1;
else
   outtmvnf = 0;
   outtmmnf = 0;
end
~                                                                                                                                                                                                           
~                                                                                                                                                                                                           
~                                                                                                                                                                                                           
~                                                                                                                                                                                                           
~                                                                                                                                                                                                           
~        
