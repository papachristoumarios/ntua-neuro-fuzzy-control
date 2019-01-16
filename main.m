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

% Choose initial condition
x0 = 0.1 * ones(length(A), 1);

% LQR Parameters
rho = 1;
Q = eye(size(A));

% Number of iterations
Niter = 200;
time = 0 : Niter;
epochs = 100;

%% Q Learning

for ep = 1 : epochs
    L = randn(size(B'));
    [H, K] = q_learning(A, B, L, Q, rho, Niter, x0);
    % Suppose we put this controller and simulate the model
    % if it is stable, we are good to go
    if max(abs(eig(A - B * K))) < 1
        break
    end
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

hold off

figure;
hold on
stem(time, - Kid * x_id');
title('Input of Ideal System')
xlabel('Samples')

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

figure;
hold on
stem(time, - K * x_q');
title('Input of Q-Learned System')
xlabel('Samples')

hold off

%% Behaviour due to random gain 

Aq_r = A - B * L;

q_learning_model = ss(Aq_r, B, zeros(size(A)), zeros(size(B)), Ts);

[~, ~, x_r] = lsim(q_learning_model, zeros(1, Niter + 1), time, x0);


figure;
hold on;
for i = 1 : length(x0)
    stem(time, x_r(:, i), 'color', rand(1,3));
end
title('Impulse responses for random Gain Matrix')
legend('x1', 'x2', 'x3')
xlabel('Samples')
ylabel('State vector')

hold off

hold off

figure;
hold on
stem(time, - L * x_r');
title('Input of System under random Gain')
xlabel('Samples')

hold off

