clc; clear; close all;

vanDerPol = @(t, y, mu) [y(2); mu * (1 - y(1)^2) * y(2) - y(1)];
mu_values = [10, 100, 1000];
eps_values = [1e-6, 1e-9, 1e-12];
tmax = 1000;
y0 = [2; 0];
solvers = {'ode45', 'ode15s'};
cpu_times = zeros(length(mu_values), length(eps_values), length(solvers));

figure('Position', [100, 100, 1200, 400]); 
for i = 1:length(mu_values)
    mu = mu_values(i);
    subplot(1, length(mu_values), i); hold on;
    title(sprintf('\\mu = %d', mu), 'FontSize', 12);
    xlabel('y_1', 'FontSize', 12); ylabel('y_2', 'FontSize', 12);
    for j = 1:length(eps_values)
        eps = eps_values(j);
        for k = 1:length(solvers)
            solver = solvers{k};
            options = odeset('RelTol', eps, 'AbsTol', eps);
            tic;
            [t, y] = feval(solver, @(t, y) vanDerPol(t, y, mu), [0, tmax], y0, options);
            cpu_times(i, j, k) = toc;
            scatter(y(:,1), y(:,2), 5, 'filled', 'DisplayName', ...
                    sprintf('%s, \\epsilon = %.0e', solver, eps));
        end
    end
    legend('show', 'Location', 'best', 'FontSize', 10);
    grid on;
end

figure('Position', [100, 100, 600, 400]); hold on;
for i = 1:length(mu_values)
    for k = 1:length(solvers)
        plot(log(eps_values), log(squeeze(cpu_times(i, :, k))), '-o', 'LineWidth', 1.5, ...
               'DisplayName', sprintf('%s, \\mu = %d', solvers{k}, mu_values(i)));
    end
end
xlabel('log(\epsilon)', 'FontSize', 12); 
ylabel('log(CPU Time)', 'FontSize', 12);
legend('show', 'Location', 'best', 'FontSize', 10);
grid on;
title('CPU Time vs. Error Tolerance', 'FontSize', 12);
