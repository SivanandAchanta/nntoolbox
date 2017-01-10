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

function [scm,alpham,hcm,ym] = fp_cpu(X,Wi,W,U,bh,bo,ff,nl,a_tanh,b_tanh,sl,lr_fw,dr_fw)


scm = (zeros(sl,nl(2)));
alpham = (zeros(sl,sl));
hcm = (zeros(sl,nl(2)));
hp = (zeros(nl(2),1));

X = X';
is = bsxfun(@plus,Wi*X, bh);

switch ff(1)
    case 'N'
        for k = 1:sl
            % forward prop
            
            ac = W*hp + is(:,k);
            sc = a_tanh*tanh(b_tanh*ac);
            
            % initiate the inner loop
            [t2,alpha_vec] = compute_fw(hcm(1:k,:), lr_fw, dr_fw, sc, k);
            ac_s = ac + t2;
            hc = a_tanh*tanh(b_tanh*ac_s);
            
            % update past hidden state
            hp = hc;
            
            % store params for T time steps
            scm(k,:) = sc';
            alpham(k,:) = [alpha_vec' zeros(1,sl-k)];
            hcm(k,:) = hc';
        end
        
        
    case 'S'
        for k = 1:sl
            % forward prop
            
            ac = W*hp + is(:,k);
            sc = 1./(1+(a_tanh*exp(-(b_tanh*ac))));
            
            % initiate the inner loop
            [t2,alpha_vec] = compute_fw(hcm(1:k,:), lr_fw, dr_fw, sc, k);
            ac_s = ac + t2;
            hc = 1./(1+(a_tanh*exp(-(b_tanh*ac_s))));
            
            % update past hidden state
            hp = hc;
            
            % store params for T time steps
            scm(k,:) = sc';
            alpham(k,:) = [alpha_vec' zeros(1,sl-k)];
            hcm(k,:) = hc';
        end
        
    case 'R' % ReLU components
        
        for k = 1:sl
            % forward prop
            
            ac = W*hp + is(:,k);
            sc = max(0,ac);
           
            % initiate the inner loop
            [t2,alpha_vec] = compute_fw(hcm(1:k,:), lr_fw, dr_fw, sc, k);
            ac_s = ac + t2;
            hc = max(0,ac_s);
            
            % update past hidden state
            hp = hc;
            
            % store params for T time steps
            scm(k,:) = sc';
            alpham(k,:) = [alpha_vec' zeros(1,sl-k)];
            hcm(k,:) = hc';
        end
        
    case 'E' % ELU components
        
        for k = 1:sl
            % forward prop
            
            ac = W*hp + is(:,k);
            sc = max(0,ac);
            sc(ac<=0) = exp(ac(ac<=0))-1;
            
            % initiate the inner loop
            [t2,alpha_vec] = compute_fw(hcm(1:k,:), lr_fw, dr_fw, sc, k);
            ac_s = ac + t2;
            hc = max(0,ac_s);
            hc(ac_s<=0) = exp(ac_s(ac_s<=0))-1;
            
            % update past hidden state
            hp = hc;
            
            % store params for T time steps
            scm(k,:) = sc';
            alpham(k,:) = [alpha_vec' zeros(1,sl-k)];
            hcm(k,:) = hc';
        end
        
    case 'P' % PELU components
        abyb = a_p./b_p;
        for k = 1:sl
            % forward prop
            
            ac = W*hp + is(:,k);
            sc = abyb.*(max(0,ac));
            ac_bp = ac./b_p;
            sc(ac<=0) = a_p(ac<=0).*(exp(ac_bp(ac<=0))-1);
            
            % initiate the inner loop
            [t2,alpha_vec] = compute_fw(hcm(1:k,:), lr_fw, dr_fw, sc, k);
            ac_s = ac + t2;
            hc = abyb.*(max(0,ac_s));
            ac_bp = ac_s./b_p;
            hc(ac_s<=0) = a_p(ac_s<=0).*(exp(ac_bp(ac_s<=0))-1);
            
            
            % update past hidden state
            hp = hc;
            
            % store params for T time steps
            scm(k,:) = sc';
            alpham(k,:) = [alpha_vec' zeros(1,sl-k)];
            hcm(k,:) = hc';
        end
        
        
    case 'M' % Softmax layer
        
        for k = 1:sl
            % forward prop
            
            ac = W*hp + is(:,k);
            intout = exp(ac);
            hc = bsxfun(@rdivide,intout,sum(intout,2));
            
            % update past hidden state
            hp = hc;
            
            % store params for T time steps
            scm(k,:) = sc';
            alpham(k,:) = [alpha_vec' zeros(1,sl-k)];
            hcm(k,:) = hc';
        end
        
    case 'L'
        for k = 1:sl
            % forward prop
            
            ac = W*hp + is(:,k);
            sc = ac;
            
            % initiate the inner loop
            [t2,alpha_vec] = compute_fw(hcm(1:k,:), lr_fw, dr_fw, sc, k);
            ac_s = ac + t2;
            hc = ac_s;
            
            % update past hidden state
            hp = hc;
            
            % store params for T time steps
            scm(k,:) = sc';
            alpham(k,:) = [alpha_vec' zeros(1,sl-k)];
            hcm(k,:) = hc';
        end
        
    otherwise
        disp('error: please enter a valid output function name (N/S/R/M/L)');
        return;
end



ac = bsxfun(@plus,U*hcm',bo);
ac = ac';

switch ff(2)
    case 'N'
        ym = a_tanh*tanh(b_tanh*ac);
    case 'S'
        ym = 1./(1+(a_tanh*exp(-(b_tanh*ac))));
    case 'R' % ReLU components
        ym = max(0,ac);
    case 'E' % ELU components
        ym = max(0,ac);
        ym(ac<=0) = exp(ac(ac<=0))-1;
    case 'M' % Softmax layer
        intout = exp(ac);
        %sum(intout,2)'
        ym = bsxfun(@rdivide,intout,sum(intout,2));
    case 'L'
        ym = ac;
    otherwise
        disp('error: please enter a valid output function name (N/S/R/M/L)');
        return;
end


end


% function[hc_ns,inp_sum] = compute_fw(hpm, lr_fw, dr_fw, hc_s, k)
%
% dr_vec = (dr_fw).^(k-(1:k));
% % dhpm = bsxfun(@times,hpm',dr_vec);
% % dhpm = hpm';
%
% % hc_s = ones(size(hc_s));
% % hc_s = (0.9.^(1:length(hc_s)))';
% inp_sum = (hpm*hc_s).*dr_vec';
% % inp_sum = (0.9.^(1:size(inp_sum)))';
% wt_sum = hpm'*inp_sum;
%
% hc_ns = lr_fw*wt_sum;
% % hc_ns = wt_sum;
%
% end
