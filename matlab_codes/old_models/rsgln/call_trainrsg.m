%{
###########################################################################
##                                                                       ##
##                                                                       ##
##                       IIIT Hyderabad, India                           ##
##                      Copyright (c) 2015                               ##
##                        All Rights Reserved.                           ##
##                                                                       ##
##  Permission is hereby granted, free of charge, to use and distribute  ##
##  this software and its documentation without restriction, including   ##
##  without limitation the rights to use, copy, modify, merge, publish,  ##
##  distribute, sublicense, and/or sell copies of this work, and to      ##
##  permit persons to whom this work is furnished to do so, subject to   ##
##  the following conditions:                                            ##
##   1. The code must retain the above copyright notice, this list of    ##
##      conditions and the following disclaimer.                         ##
##   2. Any modifications must be clearly marked as such.                ##
##   3. Original authors' names are not deleted.                         ##
##   4. The authors' names are not used to endorse or promote products   ##
##      derived from this software without specific prior written        ##
##      permission.                                                      ##
##                                                                       ##
##  IIIT HYDERABAD AND THE CONTRIBUTORS TO THIS WORK                     ##
##  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      ##
##  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   ##
##  SHALL IIIT HYDERABAD NOR THE CONTRIBUTORS BE LIABLE                  ##
##  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    ##
##  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   ##
##  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          ##
##  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       ##
##  THIS SOFTWARE.                                                       ##
##                                                                       ##
###########################################################################
##                                                                       ##
##          Author :  Sivanand Achanta (sivanand.a@research.iiit.ac.in)  ##
##          Date   :  Jul. 2015                                          ##
##                                                                       ##
###########################################################################
%}

% Load configuration file
config_basic

% Step 1 : Read data
% readdata_rnn
generate_randdata
train_test_numbats = round(train_numbats/2);

% Step 2 : Set architecture
arch_name1 = strcat(arch_name1,num2str(dout),ol_type);
arch_init
arch_name1 = strcat(arch_name1,'_',model_name,'_',wtinit_meth);

% Step 3 : Train according to SGD Type (adam/adadelta/sgdcm)
% set hyper params
switch sgd_type
    
    case 'sgdcm'
        % Training DNN using Naive SGD with classical momentum
        for l1 = l1_vec
            for l2 = l2_vec
                for lr = lr_vec
                    for mf = mf_vec
                        for gcth = gcth_vec                            
                            % Step 4 : Weight initialization
                            wt_init
                            
                            arch_name2 = strcat('_l2',num2str(l2),'_lr',num2str(lr),'_mf',num2str(mf),'_gc',num2str(gcth), ...
                                '_si',num2str(si),'_ri',num2str(ri),'_so',num2str(so),'_sy',num2str(sy),'_gt',num2str(gt_flag));
                            arch_name = strcat(arch_name1,arch_name2,'_',num2str(nwt))
                            nwt = nwt + 1;
                            
                            if gpu_flag                                
                                GWi = gpuArray(Wi);  GWfr = gpuArray(Wfr); GU = gpuArray(U); GWy = gpuArray(Wy); Gbh = gpuArray(bh); Gbo = gpuArray(bo);                                
                                GpdWi = gpuArray(zeros(size(Wi)));  GpdWfr = gpuArray(zeros(size(Wfr))); GpdU = gpuArray(zeros(size(U))); GpdWy = gpuArray(zeros(size(Wy)));
                                Gpdbh = gpuArray(zeros(size(bh))); Gpdbo = gpuArray(zeros(size(bo)));                                
                            else                                
                                GWi = Wi;  GWfr = Wfr; GU = U; GWy = (Wy); Gbh = bh; Gbo = bo; Ggn = gn; Gbn = bn;                                    
                                GpdWi = zeros(size(Wi));  GpdWfr = zeros(size(Wfr)); GpdU = zeros(size(U)); GpdWy = (zeros(size(Wy)));
                                Gpdbh = zeros(size(bh)); Gpdbo = zeros(size(bo)); Gpdgn = zeros(size(gn)); Gpdbn = zeros(size(bn));
                            end
                            
                            trainer
                            
                        end
                    end
                end
            end
        end
        
        
    case 'adadelta'
        
        % Training DNN using ADA-DELTA
        for l1 = l1_vec
            for l2 = l2_vec
                for rho = rho_vec
                    for eps_hp = eps_vec
                        for mf = mf_vec
                            for gcth = gcth_vec
                                % Step 4 : Weight initialization
                                wt_init
                                
                                arch_name2 = strcat('_l2',num2str(l2),'_rho',num2str(rho),'_eps',num2str(eps_hp),'_mf',num2str(mf),'_',num2str(gcth),...
                                    '_si',num2str(si),'_ri',num2str(ri),'_so',num2str(so),'_sy',num2str(sy),'_gt',num2str(gt_flag));
                                arch_name = strcat(arch_name1,arch_name2)
                                                               
                                % Step 5 : Training                                
                                if gpu_flag                                    
                                    GWi = gpuArray(Wi);  GWfr = gpuArray(Wfr); GU = gpuArray(U); GWy = gpuArray(Wy); Gbh = gpuArray(bh); Gbo = gpuArray(bo);                                    
                                    GpdWi = gpuArray(zeros(size(Wi)));  GpdWfr = gpuArray(zeros(size(Wfr))); GpdU = gpuArray(zeros(size(U))); GpdWy = gpuArray(zeros(size(Wy)));
                                    Gpdbh = gpuArray(zeros(size(bh))); Gpdbo = gpuArray(zeros(size(bo)));                                    
                                    GpmsgWi = gpuArray(zeros(size(Wi)));  GpmsgWfr = gpuArray(zeros(size(Wfr))); GpmsgU = gpuArray(zeros(size(U))); GpmsgWy = gpuArray(zeros(size(Wy)));
                                    Gpmsgbh = gpuArray(zeros(size(bh))); Gpmsgbo = gpuArray(zeros(size(bo)));                                    
                                    GpmsxWi = gpuArray(zeros(size(Wi)));  GpmsxWfr = gpuArray(zeros(size(Wfr))); GpmsxU = gpuArray(zeros(size(U))); GpmsxWy = gpuArray(zeros(size(Wy)));
                                    Gpmsxbh = gpuArray(zeros(size(bh))); Gpmsxbo = gpuArray(zeros(size(bo)));                                    
                                else                                    
                                    GWi = Wi;  GWfr = Wfr; GU = U; GWy = Wy; Gbh = bh; Gbo = bo; Ggn = gn; Gbn = bn;                                        
                                    GpdWi = zeros(size(Wi));  GpdWfr = zeros(size(Wfr)); GpdU = zeros(size(U)); GpdWy = (zeros(size(Wy)));
                                    Gpdbh = zeros(size(bh)); Gpdbo = zeros(size(bo)); Gpdgn = zeros(size(gn)); Gpdbn = zeros(size(bn));                                    
                                    GpmsgWi = zeros(size(Wi));  GpmsgWfr = zeros(size(Wfr)); GpmsgU = zeros(size(U)); GpmsgWy = (zeros(size(Wy)));
                                    Gpmsgbh = zeros(size(bh)); Gpmsgbo = zeros(size(bo)); Gpmsggn = zeros(size(gn)); Gpmsgbn = zeros(size(bn));                                       
                                    GpmsxWi = zeros(size(Wi));  GpmsxWfr = zeros(size(Wfr)); GpmsxU = zeros(size(U)); GpmsxWy = (zeros(size(Wy)));
                                    Gpmsxbh = zeros(size(bh)); Gpmsxbo = zeros(size(bo)); Gpmsxgn = zeros(size(gn)); Gpmsxbn = zeros(size(bn));                                       
                                end
                                
                                trainer
                                
                            end
                        end
                    end
                end
            end
        end
        
    case 'adam'
        
        % Training DNN using ADAM - SGD
        for l1 = l1_vec
            for l2 = l2_vec
                for alpha = alpha_vec
                    for beta1 = beta1_vec
                        for beta2 = beta2_vec
                            
                            % Step 4 : Weight initialization
                            wt_init
                            
                            arch_name2 = strcat('_l2',num2str(l2),'alpha',num2str(alpha),'_b1',num2str(beta1),'_b2',num2str(beta2),...
                                '_si',num2str(si),'_ri',num2str(ri),'_so',num2str(so),'_sy',num2str(sy),'_gt',num2str(gt_flag));
                            arch_name = strcat(arch_name1,arch_name2,'_',wtinit_meth)
                            
                            % Step 5 : Training
                            if gpu_flag                                
                                GWi = gpuArray(Wi);  GWfr = gpuArray(Wfr); GU = gpuArray(U); GWy = gpuArray(Wy); Gbh = gpuArray(bh); Gbo = gpuArray(bo);                                
                                GpmWi = gpuArray(zeros(size(Wi)));  GpmWfr = gpuArray(zeros(size(Wfr))); GpmU = gpuArray(zeros(size(U))); GpmWy = gpuArray(zeros(size(Wy)));
                                Gpmbh = gpuArray(zeros(size(bh))); Gpmbo = gpuArray(zeros(size(bo)));                                
                                GpvWi = gpuArray(zeros(size(Wi)));  GpvWfr = gpuArray(zeros(size(Wfr))); GpvU = gpuArray(zeros(size(U))); GpvWy = gpuArray(zeros(size(Wy)));
                                Gpvbh = gpuArray(zeros(size(bh))); Gpvbo = gpuArray(zeros(size(bo)));                                
                            else                                
                                GWi = Wi;  GWfr = Wfr; GU = U; GWy = Wy; Gbh = bh; Gbo = bo; Ggn = gn; Gbn = bn;                                    
                                GpmWi = zeros(size(Wi));  GpmWfr = zeros(size(Wfr)); GpmU = zeros(size(U)); GpmWy = (zeros(size(Wy)));
                                Gpmbh = zeros(size(bh)); Gpmbo = zeros(size(bo)); Gpmgn = zeros(size(gn)); Gpmbn = zeros(size(bn));                                  
                                GpvWi = zeros(size(Wi));  GpvWfr = zeros(size(Wfr)); GpvU = zeros(size(U)); GpvWy = (zeros(size(Wy)));
                                Gpvbh = zeros(size(bh)); Gpvbo = zeros(size(bo));  Gpvgn = zeros(size(gn)); Gpvbn = zeros(size(bn));                                     
                            end
                            
                            trainer
                            
                        end
                    end
                end
            end
        end
        
        
end



