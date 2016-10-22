function [data,targets,clv] = gen_randdata(numbats,din,dout,sl,ol_type)

data = [];
targets = [];
clv = [];

for i = 1:numbats
    
    X = randn(sl,din);
    
    if strcmp(ol_type,'L')
        Y = randn(sl,dout);
    else
        Y = zeros(sl,dout);
        for j = 1:sl
            ix = randperm(dout);
            Y(j,ix(1)) = 1;
        end
    end
    
    data = [data; X];
    targets = [targets; Y];
    clv = [clv sl];
end
clv = cumsum([1 clv]);

end