function [ H, K_opt ] = q_learning(A, B, K, Q, rho, Niter, Hprev, x0, u_samples)
   
    % Initialize variables
   
    x = x0;
    
    Y = [];
    X = [];
    
    % Policy Iteration
    for n = 1 : Niter
        u = u_samples(:, n);
        % Calculate quadratic combinations
        phi = kron([x; u], [x; u])';
        
        % Calculate reward function
        r = x' * Q * x + u' * rho * u;

        % Forward pass to the model
        x = A * x + B * u;
        t = [x; K * x];
        
        % Calculate new quadratic combinations
        
        
        r = r + t' * Hprev * t;
        
        % Append to matrices
        X = [X; phi];
        Y = [Y; r];
        
    end
    
    % Solve Least Squares Problem
    psi = pinv(X) * Y;
    
    dim = length(x) + length(u);
    H = reshape(psi, dim, dim);
    
    % Get desired elements
    Hux = H(length(x) + 1 : end, 1:length(x));
    Huu = H(length(x) + 1 : end, length(x) + 1 : end);
    
    % Policy Improvement - Find optimal value of gain
    K_opt = inv(Huu) * Hux;
    
end

