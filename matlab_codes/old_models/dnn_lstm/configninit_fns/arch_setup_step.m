if dnn_flag
    arch_init_dnn
    din = nl_dnn(end);
end

arch_name1 = strcat(arch_name1,num2str(dout),ol_type);
arch_init
arch_name1 = strcat(arch_name1,'_',model_name,'_',wtinit_meth);
