T = 17.0652165601579625588917206249;
y0 = [0.994; 0; 0; -2.00158510637908252240537862224];

options = odeset('RelTol',1e-12,'AbsTol',1e-12);
[t, y] = ode45(@arenstorf_orbit, [0 T], y0, options);

figure;
plot(y(:,1), y(:,2), 'b', 'LineWidth', 1.5);
xlabel('x');
ylabel('y');
title('Arenstorf Orbit (One Period)');
axis equal;
grid on;

Tmax = 100;
[t_long, y_long] = ode45(@arenstorf_orbit, [0 Tmax], y0, options);

figure;
plot(y_long(:,1), y_long(:,2), 'r', 'LineWidth', 1.5);
xlabel('x');
ylabel('y');
title('Arenstorf Orbit (T_{max} = 100)');
axis equal;
grid on;

solvers = {@ode45, @ode78, @ode89};
solver_names = {'ode45', 'ode78', 'ode89'};
colors = {'b', 'r', 'g'};
CPU_times = zeros(1, length(solvers));

figure;
hold on;

for i = 1:length(solvers)
    solver = solvers{i};
    tic;
    [t, y] = solver(@arenstorf_orbit, [0 Tmax], y0, options);
    CPU_times(i) = toc;
    plot(y(:,1), y(:,2), colors{i}, 'LineWidth', 1.5, 'DisplayName', solver_names{i});
end

xlabel('x');
ylabel('y');
title('Arenstorf Orbit for Different Solvers');
legend show;
axis equal;
grid on;
hold off;

fprintf('CPU Times (seconds):\n');
for i = 1:length(solvers)
    fprintf('%s: %.6f s\n', solver_names{i}, CPU_times(i));
end

function dydt = arenstorf_orbit(~, y)
    mu = 0.012277471;
    mu1 = 1 - mu;
    r1 = ((y(1) + mu)^2 + y(2)^2)^(1.5);
    r2 = ((y(1) - mu1)^2 + y(2)^2)^(1.5);
    dydt = zeros(4,1);
    dydt(1) = y(3);
    dydt(2) = y(4);
    dydt(3) = y(1) + 2*y(4) - mu1*(y(1) + mu)/r1 - mu*(y(1) - mu1)/r2;
    dydt(4) = y(2) - 2*y(3) - mu1*y(2)/r1 - mu*y(2)/r2;
end
