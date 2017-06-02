
posN = strfind(arch_name_dnn,'N');
posL = strfind(arch_name_dnn,'L');
posS = strfind(arch_name_dnn,'S');
posR = strfind(arch_name_dnn,'R');
posM = strfind(arch_name_dnn,'M');
posE = strfind(arch_name_dnn,'E');
posP = strfind(arch_name_dnn,'P');
posfull = sort([posN,posL,posS,posR,posM,posE,posP]);
posfull = [0 posfull];

nl_dnn = [din];
for i = 1:length(posfull)-1
    nl_dnn = [nl_dnn str2num(arch_name_dnn(posfull(i)+1:posfull(i+1)-1))];
    f_dnn(i) = arch_name_dnn(posfull(i+1));    
end

% Do Not Change The Following Variables
nh_dnn = length(nl_dnn) - 1; % number of hidden layers

if (length(nl_dnn) - 1) ~= length(f_dnn)
    disp('number of hidden o/p fns mus be same as number of hidden layers');
end

nlv_dnn = 1:nh_dnn;
wtl_dnn = [1 nl_dnn(nlv_dnn).*nl_dnn(nlv_dnn+1)];
wtl_dnn = cumsum(wtl_dnn);
btl_dnn = cumsum([1 nl_dnn(nlv_dnn+1)]);

arch_name_dnn = strcat(num2str(din),'L');
for i = 1:nh_dnn
    arch_name_dnn = strcat(arch_name_dnn,num2str(nl_dnn(i+1)),f_dnn(i));
end
arch_name_dnn