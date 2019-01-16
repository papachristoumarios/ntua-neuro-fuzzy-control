function [ H, K ] = q_learning(A, B, K, Q, rho, Niter, x0)
   
    % Initialize variables
    x = x0;
    u = -K * x;
    Y = [];
    X = [];
   
    for n = 1 : Niter
        
        % Calculate quadratic combinations
        phi = quad_comb(x, u);
        r = x' * Q * x + u' * rho * u;

        % update x_k+1 u_k + 1
        x = A * x + B * u;
        u = -K * x;

        % Calculate new quadratic combinations
        phi_new = quad_comb(x, u);
        
        % Calculate entry
        dphi = phi' - phi_new';
        
        % Append to matrices
        X = [X; dphi];
        Y = [Y; r];
        
    end
    
    % Solve Least Squares Problem
    psi = pinv(X) * Y;
    
    dim = length(x) + length(u);
    H = reshape(psi, dim, dim);
    
    % Get desired elements
    Hux = H(length(x) + 1 : end, 1:length(x));
    Huu = H(length(x) + 1 : end, length(x) + 1 : end);
    
    % Find optimal value of gain
    K = inv(Huu) * Hux;
    
end

