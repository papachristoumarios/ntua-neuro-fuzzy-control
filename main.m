% Q Learning for Optimal Control with LQR Criterion
% Author: Marios Papachristou
% AM: 03115101
% email: papachristoumarios@gmail.com

%% Parameters
% Actual System Dynamics
A = [0 1 0; 0 0 1; 0 0 0];
B = [0; 0; 1];
dim = size(A, 1) + size(B, 2);


% Sampling Period
Ts = 1;
sys = ss(A, B, zeros(size(A)), zeros(size(B)), Ts);

% Choose initial condition
x0 = 0.1 * ones(length(A), 1);

% LQR Parameters
rho = 1;
Q = eye(size(A));

%% Ideal LQR
% Solution to DARE
[Kid, Pid, e] = dlqr(A, B, Q, rho);

% Ideal H matrix
Hid = [Q + A' * Pid * A A' * Pid * B;
       B' * Pid * A rho + B' * Pid * B];


% Number of iterations
Niter = 200;
time = 0 : Niter - 1;
epochs = 4;

gain_norms = zeros(epochs, 1);
H_norms = zeros(epochs, 1);

%% Q Learning
L = randn(size(B'));
Hp = zeros(dim, dim);
for ep = 1 : epochs
    
    % Take random inputs
    u_samples = randn(size(B, 2), Niter);

    [H, K] = q_learning(A, B, L, Q, rho, Niter, Hp, x0, u_samples);

    L = K;
    Hp = H;
    
    % Update norms plot
    gain_norms(ep) = norm(L - Kid, 'fro');
    H_norms(ep) = norm(H - Hid, 'fro');
    
    % Plot results for epoch
    Aq_c = A - B * K;

    q_learning_model = ss(Aq_c, zeros(size(B)), zeros(size(A)), zeros(size(B)), Ts);

    [~, ~, x_q] = lsim(q_learning_model, zeros(1, Niter), time, x0);

    figure;
    for i = 1 : length(x0)
        subplot(2, 2, i);
        stem(time, x_q(:, i), 'color', rand(1,3));
        title(sprintf('Impulse responses for Q-Learned system for epoch %d', ep));
        xlabel('Samples');
        ylabel(sprintf('x_%d', i));
    end
    
    subplot(2,2,4);
    stem(time, - K * x_q');
    title('Input of Q-Learned System for the new policy');
    xlabel('Samples');
    ylabel('u');

    % Simulate response under random noise
    
    [~, ~, x_n] = lsim(sys, u_samples, time);
    figure;
    for i = 1 : length(x0)
        subplot(2, 2, i);
        stem(time, x_n(:, i), 'color', rand(1,3));
        title(sprintf('Response for random input (Epoch %d)', ep));
        xlabel('Samples');
        ylabel(sprintf('x_%d', i));
    end
    
    subplot(2,2,4);
    stem(time, u_samples);
    title('Input');
    xlabel('Samples');
    ylabel('u');
    
end

%% Plot norm differences

figure;
plotyy(1:epochs, H_norms, 1 : epochs, gain_norms);
title('Norms difference from ideal (Frobenius)');
legend('|| H - H* ||', '|| K - K* ||');
xlabel('Epochs');


%% Ideal LQR

Aid_c = A - B * Kid;

ideal_model = ss(Aid_c, zeros(size(B)), zeros(size(A)), zeros(size(B)), Ts);
[~, ~, x_id] = lsim(ideal_model, zeros(1, Niter), time, x0);


figure;
hold on;
for i = 1 : length(x0)
    stem(time, x_id(:, i), 'color', rand(1,3));
end
title('Impulse responses for ideal system')
legend('x1', 'x2', 'x3')
xlabel('Samples')
ylabel('State vector')

hold off

hold off

figure;
hold on
stem(time, - Kid * x_id');
title('Input of Ideal System')
xlabel('Samples')

hold off






