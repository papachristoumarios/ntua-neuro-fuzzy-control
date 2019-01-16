function [ H, K ] = q_learning(A, B, K, Q, rho, Niter, x0)
   
    x = x0;
    u = -K * x;
    Y = [];
    X = [];
   
    for n = 1 : Niter
        
        phi = quad_comb(x, u);
        r = x' * Q * x + u' * rho * u;

        % update x_k+1 u_k + 1
        x = A * x + B * u;
        u = -K * x;

        phi_new = quad_comb(x, u);
        
        dphi = phi' - phi_new';
        
        X = [X; dphi];
        Y = [Y; r];
        
    end

    psi = pinv(X) * Y;
    
    dim = length(x) + length(u);
    H = reshape(psi, dim, dim);
    
    Hux = H(length(x) + 1 : end, 1:length(x));
    Huu = H(length(x) + 1 : end, length(x) + 1 : end);
    
    K = inv(Huu) * Hux;
    
end

