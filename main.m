% Q Learning for Optimal Control with LQR Criterion
% Author: Marios Papachristou
% AM: 03115101
% email: papachristoumarios@gmail.com

%% Parameters
% Actual System Dynamics
A = [0 1 0; 0 0 1; 0 0 0];
B = [0; 0; 1];

% Sampling Period
Ts = 1;


% Choose a random gain L such that A - BL is stable
L = randn(size(B'));
while max(abs(A - B * L)) >= 1
    L = randn(size(B'));
end


% Choose initial condition
x0 = 0.1 * ones(length(A), 1);

% LQR Parameters
rho = 1;
Q = eye(size(A));

% Number of iterations
Niter = 200;
time = 0 : Niter;
epochs = 1;

%% Q Learning
for ep = 1 : epochs
    [H, K] = q_learning(A, B, L, Q, rho, Niter, x0);
    L = K;
end
    
%% Ideal LQR
[Kid, Pid, e] = dlqr(A, B, Q, rho);

Aid_c = A - B * Kid;

ideal_model = ss(Aid_c, B, zeros(size(A)), zeros(size(B)), Ts);

[~, ~, x_id] = lsim(ideal_model, zeros(1, Niter + 1), time, x0);


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

%% Q-Learned LQR

Aq_c = A - B * K;

q_learning_model = ss(Aq_c, B, zeros(size(A)), zeros(size(B)), Ts);

[~, ~, x_q] = lsim(q_learning_model, zeros(1, Niter + 1), time, x0);


figure;
hold on;
for i = 1 : length(x0)
    stem(time, x_q(:, i), 'color', rand(1,3));
end
title('Impulse responses for Q-Learned system')
legend('x1', 'x2', 'x3')
xlabel('Samples')
ylabel('State vector')

hold off

%% Compare MMSE
MSE = mean(std(x_q - x_id));
display(sprintf('MMSE between state vectors: %f', MSE));
