function [avg_loss] = compute_meansquarederror(P,T)

avg_loss = mean(sum((T - P).^2,2));

end

