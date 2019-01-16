function [Z] = quad_comb(x, u)
    t = [x ; u];
    Z = reshape(t * t', length(t)^2 , 1);
end