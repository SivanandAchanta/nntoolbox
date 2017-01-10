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

function [hcm,ym] = fp_cpu_test(X,Wi,Wfr,U,Ufr,bh,bo,ff,nl,a_tanh,b_tanh,sl)

hcm = (zeros(sl,nl(2)));
hp = (zeros(nl(2),1));

X = X';
is = bsxfun(@plus,Wi*X, bh);

switch ff(1)
    case 'N'
        for k = 1:sl
            % forward prop
            ac = Wfr*hp + is(:,k);
            hc = a_tanh*tanh(b_tanh*ac);
            hp = hc;
            % store params for T time steps
            hcm(k,:) = hc';
        end
        
        
    case 'S'
        for k = 1:sl
            % forward prop
            ac = Wfr*hp + is(:,k);
            hc = 1./(1+(a_tanh*exp(-(b_tanh*ac))));
            hp = hc;
            % store params for T time steps
            hcm(k,:) = hc';
        end
        
    case 'R' % ReLU components
        
        for k = 1:sl
            % forward prop
            ac = Wfr*hp + is(:,k);
            hc = max(0,ac);
            hp = hc;
            % store params for T time steps
            hcm(k,:) = hc';
        end

    case 'E' % ELU components

        for k = 1:sl
            % forward prop
            ac = Wfr*hp + is(:,k);
            hc = max(0,ac);
            hc(ac<=0) = exp(ac(ac<=0))-1; 
            hp = hc;
            % store params for T time steps
            hcm(k,:) = hc';
        end

    case 'P' % PELU components
        abyb = a_p./b_p;  
        for k = 1:sl
            % forward prop
            ac = Wfr*hp + is(:,k);
            hc = abyb.*(max(0,ac));
            ac_bp = ac./b_p;
            hc(ac<=0) = a_p(ac<=0).*(exp(ac_bp(ac<=0))-1);
            hp = hc;
            % store params for T time steps
            hcm(k,:) = hc';
        end

    
    case 'M' % Softmax layer
        
        for k = 1:sl
            % forward prop
            ac = Wfr*hp + is(:,k);
            intout = exp(ac);
            hc = bsxfun(@rdivide,intout,sum(intout,2));
            hp = hc;
            % store params for T time steps
            hcm(k,:) = hc';
        end
        
    case 'L'
        for k = 1:sl
            % forward prop
            ac = Wfr*hp + is(:,k);
            hc = ac;
            hp = hc;
            % store params for T time steps
            hcm(k,:) = hc';
        end
        
    otherwise
        disp('error: please enter a valid output function name (N/S/R/M/L)');
        return;
end


ym = (zeros(sl,nl(3)));
yp = (zeros(nl(3),1));

is = bsxfun(@plus,U*hcm',bo);

switch ff(2)
    case 'N'
        for k = 1:sl
            % forward prop
            ac = Ufr*yp + is(:,k);
            yc = a_tanh*tanh(b_tanh*ac);
            yp = yc;
            % store params for T time steps
            ym(k,:) = yc';
        end
        
        
    case 'S'
        for k = 1:sl
            % forward prop
            ac = Ufr*yp + is(:,k);
            yc = 1./(1+(a_tanh*exp(-(b_tanh*ac))));
            yp = yc;
            % store params for T time steps
            ym(k,:) = yc';
        end
        
    case 'R' % ReLU components
        
        for k = 1:sl
            % forward prop
            ac = Ufr*yp + is(:,k);
            yc = max(0,ac);
            yp = yc;
            % store params for T time steps
            ym(k,:) = yc';
        end

    case 'E' % ELU components

        for k = 1:sl
            % forward prop
            ac = Ufr*yp + is(:,k);
            yc = max(0,ac);
            yc(ac<=0) = exp(ac(ac<=0))-1; 
            yp = yc;
            % store params for T time steps
            ym(k,:) = yc';
        end

    case 'P' % PELU components
        abyb = a_p./b_p;  
        for k = 1:sl
            % forward prop
            ac = Ufr*yp + is(:,k);
            yc = abyb.*(max(0,ac));
            ac_bp = ac./b_p;
            yc(ac<=0) = a_p(ac<=0).*(exp(ac_bp(ac<=0))-1);
            yp = yc;
            % store params for T time steps
            ym(k,:) = yc';
        end

    
    case 'M' % Softmax layer
        
        for k = 1:sl
            % forward prop
            ac = Ufr*yp + is(:,k);
            intout = exp(ac);
            yc = bsxfun(@rdivide,intout,sum(intout,2));
            yp = yc;
            % store params for T time steps
            ym(k,:) = yc';
        end
        
    case 'L'
        for k = 1:sl
            % forward prop
            ac = Ufr*yp + is(:,k);
            yc = ac;
            yp = yc;
            % store params for T time steps
            ym(k,:) = yc';
        end
        
    otherwise
        disp('error: please enter a valid output function name (N/S/R/M/L)');
        return;
end


end
