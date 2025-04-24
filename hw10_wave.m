close all; clear;

a = sqrt(2);
h = 0.05;
k = h / a;
x = -6:h:6;
N = length(x);
time_factors = [0.5, 1, 2, 4];
schemes = {
    {@RightBW, @LeftBW, 'Beam-Warming'}, ...
    {@UpwindRight, @UpwindLeft, 'Upwind'}, ...
    {@RightBW, @LaxF, 'Lax-Friedrichs'}, ...
    {@RightBW, @LaxW, 'Lax-Wendroff'}
};
for s = 1:length(schemes)
    xi_solver = schemes{s}{1};
    eta_solver = schemes{s}{2};
    method_name = schemes{s}{3};
    figure;
    for ti = 1:length(time_factors)
        d = time_factors(ti);
        T = d / a;
        tspan = 0:k:T;
        xi = zeros(N, 1);
        eta = zeros(N, 1);
        for i = 1:N
            xi(i) = (-tsi(x(i)) + a * dphi(x(i))) / (2 * a);
            eta(i) = (tsi(x(i)) + a * dphi(x(i))) / (2 * a);
        end
        xi_out = xi_solver(xi, -a, k, h, tspan);
        eta_out = eta_solver(eta, a, k, h, tspan);
        y = [xi_out'; eta_out'];
        w = [a, -a; 1, 1] * y;
        ux = w(2, :)';
        u = zeros(N, 1);
        for i = 2:N
            u(i) = u(i-1) + h * ux(i-1);
        end
        exact = zeros(N, 1);
        for i = 1:N
            exact(i) = 0.5 * (phi(x(i) + a*T) + phi(x(i) - a*T));
        end
        subplot(2, 2, ti);
        plot(x, u, 'b', x, exact, 'k--', 'LineWidth', 1.5);
        title([method_name, ', T = ', num2str(T)]);
        legend('Numerical', 'Exact');
        xlabel('x'); ylabel('u(x, T)');
    end
    sgtitle(['Comparison using ', method_name]);
    saveas(gcf, ['Fig_' strrep(method_name, ' ', '_') '.png']);
end

function f = phi(x)
f = max(1-abs(x),0);
end
function f = dphi(x)
if (-1<x) && (x<0)
    f=1;
elseif (0<x) && (x<1)
    f=-1;
else
    f=0;
end
end
function f = tsi(x)
f=0;
end
function u = LaxF(u,a,k,h,t)
N = size(u,1);
unew = u;
for i = 2:length(t)
    for j = 1:N
        if j == 1
            unew(j,1) = (u(2,1)+u(N,1))/2 - a*k/2/h*(u(2,1)-u(N,1));
        elseif j == N
            unew(j,1) = (u(1,1)+u(N-1,1))/2 - a*k/2/h*(u(1,1)-u(N-1,1));
        else
            unew(j,1) = (u(j+1,1)+u(j-1,1))/2 - a*k/2/h*(u(j+1,1)-u(j-1,1));
        end
    end
    u = unew;
end
end
function u = LaxW(u,a,k,h,t)
N = size(u,1);
unew = u;
for i = 1:length(t)-1
    for j = 1:N
        if j == 1
            unew(j,1) = u(1,1) - a*k/2/h*(u(2,1)-u(N,1)) + (a*k/h)^2/2*(u(2,1)-2*u(1,1)+u(N,1));
        elseif j == N
            unew(j,1) = u(N,1) - a*k/2/h*(u(1,1)-u(N-1,1)) + (a*k/h)^2/2*(u(1,1)-2*u(N,1)+u(N-1,1));
        else
            unew(j,1) = u(j,1) - a*k/2/h*(u(j+1,1)-u(j-1,1)) + (a*k/h)^2/2*(u(j+1,1)-2*u(j,1)+u(j-1,1));
        end
    end
    u = unew;
end
end
function u = UpwindLeft(u,a,k,h,t)
N = size(u,1);
unew = u;
for i = 1:length(t)-1
    for j = 1:N
        if j == 1
            unew(j,1) = u(j,1) - a*k/h*(u(j,1)-u(N,1));
        else
            unew(j,1) = u(j,1) - a*k/h*(u(j,1)-u(j-1,1));
        end
    end
    u = unew;
end
end
function u = UpwindRight(u,a,k,h,t)
N = size(u,1);
unew = u;
for i = 1:length(t)-1
    for j = 1:N
        if j == N
            unew(j,1) = u(j,1) - a*k/h*(u(1,1)-u(j,1));
        else
            unew(j,1) = u(j,1) - a*k/h*(u(j+1,1)-u(j,1));
        end
    end
    u = unew;
end
end
function u = LeftBW(u,a,k,h,t)
N = size(u,1);
unew = u;
for i = 1:length(t)-1
    for j = 1:N
        if j == 1
            unew(j,1) = u(j,1) - a*k/2/h*(3*u(j,1)-4*u(N,1)+u(N-1,1)) + (a*k/h)^2/2*(u(j,1)-2*u(N,1)+u(N-1,1));
        elseif j == 2
            unew(j,1) = u(j,1) - a*k/2/h*(3*u(j,1)-4*u(j-1,1)+u(N,1)) + (a*k/h)^2/2*(u(j,1)-2*u(j-1,1)+u(N,1));
        else
            unew(j,1) = u(j,1) - a*k/2/h*(3*u(j,1)-4*u(j-1,1)+u(j-2,1)) + (a*k/h)^2/2*(u(j,1)-2*u(j-1,1)+u(j-2,1));
        end
    end
    u = unew;
end
end
function u = RightBW(u,a,k,h,t)
N = size(u,1);
unew = u;
for i = 1:length(t)-1
    for j = 1:N
        if j == N
            unew(j,1) = u(j,1) - a*k/2/h*(-3*u(j,1)+4*u(1,1)-u(2,1)) + (a*k/h)^2/2*(u(j,1)-2*u(1,1)+u(2,1));
        elseif j == N-1
            unew(j,1) = u(j,1) - a*k/2/h*(-3*u(j,1)+4*u(j+1,1)-u(1,1)) + (a*k/h)^2/2*(u(j,1)-2*u(j+1,1)+u(1,1));
        else
            unew(j,1) = u(j,1) - a*k/2/h*(-3*u(j,1)+4*u(j+1,1)-u(j+2,1)) + (a*k/h)^2/2*(u(j,1)-2*u(j+1,1)+u(j+2,1));
        end
    end
    u = unew;
end
end