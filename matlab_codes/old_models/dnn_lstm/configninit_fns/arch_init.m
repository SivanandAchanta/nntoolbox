
posN = strfind(arch_name1,'N');
posL = strfind(arch_name1,'L');
posS = strfind(arch_name1,'S');
posR = strfind(arch_name1,'R');
posM = strfind(arch_name1,'M');
posE = strfind(arch_name1,'E');
posP = strfind(arch_name1,'P');
posfull = sort([posN,posL,posS,posR,posM,posE,posP]);
posfull = [0 posfull];

nl = [din];
for i = 1:length(posfull)-1
    nl = [nl str2num(arch_name1(posfull(i)+1:posfull(i+1)-1))];
    f(i) = arch_name1(posfull(i+1));    
end

% Do Not Change The Following Variables
nh = length(nl) - 1; % number of hidden layers

if (length(nl) - 1) ~= length(f)
    disp('number of hidden o/p fns mus be same as number of hidden layers');
end

arch_name1 = strcat(num2str(din),'L');
for i = 1:nh
    arch_name1 = strcat(arch_name1,num2str(nl(i+1)),f(i));
end
arch_name1