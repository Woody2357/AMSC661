function generate_figures()
    n = 100;
    h = 2 / (n + 1);
    xi = linspace(-1, 1, n+2)';
    tspan = [0 1.2];
    t_query = 0.1:0.1:1.2;
    opts = odeset('RelTol',1e-6,'AbsTol',1e-8);
    ICs = {@(x) max(0, 1 - x.^2), ...
           @(x) (abs(x)<1) .* (1 - 0.99*cos(2*pi*x))};
    for ic_idx = 1:2
        x_f0 = 1.0;
        x0_0 = 0;
        x_L0 = x0_0 - x_f0;
        x_R0 = x0_0 + x_f0;
        x_init = x0_0 + x_f0 * xi;
        u0 = ICs{ic_idx}(x_init);
        uvec0 = [u0(2:end-1); x_L0; x_R0];
        [tvals, u_all] = ode15s(@(t,u) rhs(t, u, n, h), tspan, uvec0, opts);
        u_interp = interp1(tvals, u_all, t_query);
        X_all = zeros(n+2, length(t_query));
        U_all = zeros(n+2, length(t_query));
        U_norm_all = zeros(n+2, length(t_query));
        for k = 1:length(t_query)
            uvec = u_interp(k,:)';
            u_inner = uvec(1:n);
            x_L = uvec(n+1);
            x_R = uvec(n+2);
            x0 = (x_L + x_R)/2;
            x_f = (x_R - x_L)/2;
            u_full = [0; u_inner; 0];
            x = x0 + x_f * xi;
            X_all(:,k) = x;
            U_all(:,k) = u_full;
            U_norm_all(:,k) = u_full / max(u_full);
        end
        figure;
        plot(X_all, U_all);
        title(sprintf('u(x,t) for IC %d', ic_idx));
        xlabel('x'); ylabel('u(x,t)');
        legend(arrayfun(@(t) sprintf('t=%.1f', t), t_query, 'UniformOutput', false));
        xlim([-2 2]); ylim([0 1]);
        figure;
        plot(xi, U_norm_all);
        hold on;
        plot(xi, 1 - xi.^2, 'r-', 'LineWidth', 2, 'DisplayName','1 - \xi^2');
        hold off;
        title(sprintf('u(\\xi,t)/max for IC %d', ic_idx));
        xlabel('\xi'); ylabel('u(\xi,t)/max');
        legend(arrayfun(@(t) sprintf('t=%.1f', t), t_query, 'UniformOutput', false));
        ylim([0 1.2]);
    end
end


function dudt = rhs(~, uvec, n, h)
    u = uvec(1:n);
    x_L = uvec(n+1);
    x_R = uvec(n+2);
    x0 = (x_L + x_R)/2;
    x_f = (x_R - x_L)/2;
    u_ext = [0; u; 0];
    xi = linspace(-1, 1, n+2)';
    xi_inner = xi(2:end-1);
    ux = zeros(n, 1);
    uxx = zeros(n, 1);
    for i = 1:n
        ux(i) = (u_ext(i+2) - u_ext(i)) / (2*h);
        uxx(i) = (u_ext(i+2) - 2*u_ext(i+1) + u_ext(i)) / h^2;
    end
    duL = (4*u_ext(2) - u_ext(3)) / (2*h);    % at xi = -1
    duR = (-4*u_ext(end-1) + u_ext(end-2)) / (2*h);  % at xi = 1
    term1 = -0.5 * ((1 + xi_inner) * duR + (1 - xi_inner) * duL) .* ux;
    term2 = u .* uxx;
    term3 = ux.^2;
    dudt_inner = (term1 + term2 + term3) / x_f^2;
    dxL = -duL / x_f;
    dxR = -duR / x_f;
    dudt = [dudt_inner; dxL; dxR];
end
