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

function [scm,hcm,ym] = fp_cpu(X,Wi,W,U,bh,bo,ff,nl,a_tanh,b_tanh,sl)

hcm = (zeros(sl,nl(2)));
scm = (zeros(sl,nl(2)));
hp = (zeros(nl(2),1));

X = X';
is = bsxfun(@plus,Wi*X, bh);

% alpha = 0.7;
% beta = 1 - alpha;
% alpha_vec = alpha*ones(nl(2),1);
% beta_vec = beta*ones(nl(2),1);
% alpha_vec = linspace(0,alpha,nl(2))';
% beta_vec = 1 - alpha_vec;

switch ff(1)
    case 'N'
        for k = 1:sl
            % forward prop
            ac = W*hp + is(:,k);
            sc = a_tanh*tanh(b_tanh*ac);
            
            %hc = alpha_vec.*sc + beta_vec.*hp;
            hc = sc + hp;
            hp = hc;
            
            % store params for T time steps
            hcm(k,:) = hc';
            scm(k,:) = sc';
        end
        
        
    case 'S'
        for k = 1:sl
            % forward prop
            ac = W*hp + is(:,k);
            sc = 1./(1+(a_tanh*exp(-(b_tanh*ac))));
            
            %             hc = alpha_vec.*sc + beta_vec.*hp;
            hc = sc + hp;
            hp = hc;
            
            % store params for T time steps
            hcm(k,:) = hc';
            scm(k,:) = sc';
        end
        
    case 'R' % ReLU components
        
        for k = 1:sl
            % forward prop
            ac = W*hp + is(:,k);
            sc = max(0,ac);
            
            %             hc = alpha_vec.*sc + beta_vec.*hp;
            hc = sc + hp;
            hp = hc;
            
            % store params for T time steps
            hcm(k,:) = hc';
            scm(k,:) = sc';
        end
        
    case 'E' % ELU components
        
        for k = 1:sl
            % forward prop
            ac = W*hp + is(:,k);
            sc = max(0,ac);
            sc(ac<=0) = exp(ac(ac<=0))-1;
            
            %             hc = alpha_vec.*sc + beta_vec.*hp;
            hc = sc + hp;
            hp = hc;
            
            % store params for T time steps
            hcm(k,:) = hc';
            scm(k,:) = sc';
        end
        
    case 'P' % PELU components
        abyb = a_p./b_p;
        for k = 1:sl
            % forward prop
            ac = W*hp + is(:,k);
            sc = abyb.*(max(0,ac));
            ac_bp = ac./b_p;
            sc(ac<=0) = a_p(ac<=0).*(exp(ac_bp(ac<=0))-1);
            
            %             hc = alpha_vec.*sc + beta_vec.*hp;
            hc = sc + hp;
            hp = hc;
            
            % store params for T time steps
            hcm(k,:) = hc';
            scm(k,:) = sc';
        end
        
        
    case 'M' % Softmax layer
        
        for k = 1:sl
            % forward prop
            ac = W*hp + is(:,k);
            intout = exp(ac);
            hc = bsxfun(@rdivide,intout,sum(intout,2));
            hp = hc;
            % store params for T time steps
            hcm(k,:) = hc';
        end
        
    case 'L'
        for k = 1:sl
            % forward prop
            ac = W*hp + is(:,k);
            sc = ac;
            
            %             hc = alpha_vec.*sc + beta_vec.*hp;
            hc = sc + hp;
            hp = hc;
            
            % store params for T time steps
            hcm(k,:) = hc';
            scm(k,:) = sc';
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
