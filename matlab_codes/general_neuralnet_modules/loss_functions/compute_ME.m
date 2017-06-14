function [avg_loss] = compute_ME(P,T)

avg_loss = mean(sum(abs(T - P),2));

end

