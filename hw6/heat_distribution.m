n = 100;
xmin = -pi;
xmax = pi;
ymin = 0;
ymax = 2;
hx = (xmax-xmin)/n;
hy = (ymax-ymin)/n;
x = xmin:hx:xmax;
y = ymin:hy:ymax;
[yg,xg] = meshgrid(y,x);
e = ones(n,1);
Lx = spdiags([e,e,-2*e,e,e],[-n+1,-1,0,1,n-1],n,n);
Ly = spdiags([e,-2*e,e],[-1,0,1],n,n);
Ly(1,2) = 2;
f = zeros(n);
for i = 1:n
    xi=x(i);
    if (xi>=-pi/2 && xi<=pi/2)
        f(i,:)=-cos(xi);
    end
end
f = reshape(f,n^2,1);
I = eye(n);
A = 1/hx^2*kron(I,Lx) + 1/hy^2*kron(Ly,I);
u = zeros(n+1,n+1);
u(1:n,1:n) = reshape(A\f,n,n);
u(n+1,1:n)=u(1,1:n);

figure(1);
hold on; grid on;
contourf(xg,yg,u,linspace(min(min(u)),max(max(u)),20));
xlabel('x');
ylabel('y');
hbar = colorbar;
