clc; clear; close all;

nx = 100; ny = 160;
x = linspace(-4,1,nx);
y = linspace(-4,4,ny);
[X, Y] = meshgrid(x, y);
Z = X + 1i*Y;

R_FE = 1 + Z;
R_Mid = 1 + Z + (1/2)*Z.^2;
R_Kutta = 1 + Z + (1/2)*Z.^2 + (1/6)*Z.^3;
R_RK4 = 1 + Z + (1/2)*Z.^2 + (1/6)*Z.^3 + (1/24)*Z.^4;

A = zeros(7,7);
A(2,1) = 1/5;
A(3,1:2) = [3/40,9/40];
A(4,1:3) = [44/45,-56/15,32/9];
A(5,1:4) = [19372/6561,-25360/2187,64448/6561,-212/729];
A(6,1:5) = [9017/3168,-355/33,46732/5247,49/176,-5103/18656];
A(7,1:6) = [35/384,0,500/1113,125/192,-2187/6784,11/84];
y = [35/384,0,500/1113,125/192,-2187/6784,11/84,0];
y = [5179/57600,0,7571/16695,393/640,-92097/339200,187/2100,0];
e = ones(7,1);
R_DOPRI5 = 1 + Z.*(y*e + y*A*e*Z + y*A^2*e*Z.^2 + y*A^3*e*Z.^3 + y*A^4*e*Z.^4 + y*A^5*e*Z.^5 + y*A^6*e*Z.^6);

figure;
methods = {'Forward Euler', 'Midpoint Rule', 'Kutta''s 3rd Order', 'RK4', 'DOPRI5(4)'};
R_methods = {R_FE, R_Mid, R_Kutta, R_RK4, R_DOPRI5};

for i = 1:length(R_methods)
    subplot(2,3,i);
    contourf(X, Y, abs(R_methods{i}), [1 1], 'b');
    hold on;
    contour(X, Y, abs(R_methods{i}), [1 1], 'k', 'LineWidth', 2);
    title(methods{i});
    xlabel('Re(z)');
    ylabel('Im(z)');
    axis equal;
    grid on;
end

sgtitle('Absolute Stability Regions of ERK Methods');

