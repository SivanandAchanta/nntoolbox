function [otl] = get_otl(bs,nl,nlv)
% train set variables
otl = [1 bs*(nl(nlv+1))];
otl = cumsum(otl);
