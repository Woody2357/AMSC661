%%
close all;
addpath('./distmesh');
fd = @(p) ddiff(dcircle(p,0,0,2),dcircle(p,0,0,1));
[pts,tri]=distmesh2d(fd,@huniform,0.1,[-2,-2;2,2],[]);
eps=1e-4;
distance=sqrt(pts(:,1).^2+pts(:,2).^2);
dirichlet1=find(abs(distance-1)<eps);
dirichlet2=find(abs(distance-2)<eps);
dirichlet=[dirichlet1;dirichlet2];


% FEM2D_HEAT finite element method for two-dimensional heat equation. 
% Initialization
Npts = size(pts,1);
Ntri = size(tri,1);
FreeNodes=setdiff(1:Npts,unique(dirichlet));
A = sparse(Npts,Npts);
B = sparse(Npts,Npts); 
T = 1; dt = 0.01; N = T/dt;
U = zeros(Npts,N+1);
% Assembly
for j = 1:Ntri
	A(tri(j,:),tri(j,:)) = A(tri(j,:),tri(j,:)) + stima3(pts(tri(j,:),:));
end
for j = 1:Ntri
	B(tri(j,:),tri(j,:)) = B(tri(j,:),tri(j,:)) + ...
        det([1,1,1;pts(tri(j,:),:)'])*[2,1,1;1,2,1;1,1,2]/24;
end
% Initial Condition
U(:,1) = IC(pts); 
% time steps
for n = 2:N+1
    b = sparse(Npts,1);
    % Volume Forces
    for j = 1:Ntri
        b(tri(j,:)) = b(tri(j,:)) + ... 
            det([1,1,1; pts(tri(j,:),:)']) * ... 
            dt*myf(sum(pts(tri(j,:),:))/3,n*dt)/6;
    end
    % Neumann conditions
    % for j = 1 : size(neumann,1)
    %    b(neumann(j,:)) = b(neumann(j,:)) + ...
    %      norm(pts(neumann(j,1),:)-pts(neumann(j,2),:))*...
    %      dt*myg(sum(pts(neumann(j,:),:))/2,n*dt)/2;
    % end
    % previous timestep
    % b=b+B*U(:,n-1);
    b=b+(-1/2*dt*A+B)*U(:,n-1);
    % Dirichlet conditions
    u = sparse(Npts,1);
    u(unique(dirichlet)) = myu_d(pts(unique(dirichlet),:),n*dt);
    % b=b-(dt*A+B)*u;
    b=b-(1/2*dt*A+B)*u;
    % Computation of the solution
    u(FreeNodes) = (1/2*dt*A(FreeNodes,FreeNodes)+ ...
            B(FreeNodes,FreeNodes))\b(FreeNodes);
    U(:,n) = u;
    t = n*dt;

if t == 0.1
    figure;
    trisurf(tri,pts(:,1),pts(:,2),full(u)','facecolor','interp')
    title(sprintf('Time = %.1f\n',t),'Fontsize',14);
    axis ij
    colorbar
    view(2)
    set(gca,'Fontsize',14);
end
if t == 1
    figure;
    trisurf(tri,pts(:,1),pts(:,2),full(u)','facecolor','interp')
    title(sprintf('Time = %.1f\n',t),'Fontsize',14);
    axis ij
    colorbar
    view(2)
    set(gca,'Fontsize',14);
    figure;
    r=sqrt(pts(:,1).^2+pts(:,2).^2);
    [rsort,isort] = sort(r,'ascend');
    usort = u(isort);
    plot(rsort,usort,'LineWidth',2);
    hold on
    true = (1-rsort.^2)/4+3*log(rsort)/4/log(2);
    plot(rsort,true,'LineWidth',2);
    legend('numerical','exact');
end
end

% figure
% for k = 1 : 6
%     subplot(2,3,k)
%     t = 0.2*(k - 1);
%     p = ceil(t/dt) + 1;
%     u = U(:,p);
%     trisurf(tri,pts(:,1),pts(:,2),full(u)','facecolor','interp')
%     title(sprintf('Time = %.1f\n',t),'Fontsize',14);
%     axis ij
%     colorbar
%     view(2)
%     set(gca,'Fontsize',14);
% end
% end
%%
function u0 = IC(x)
r = sqrt(x(:,1).^2 + x(:,2).^2);
u0 = r + x(:,1)./r;
end

%%
function DirichletBoundaryValue = myu_d(x,t)
% xmin = min(x(:,1));
% xmax = max(x(:,1));
% midx = 0.5*(xmin + xmax);
% DirichletBoundaryValue =  0.5 * (sign(x(:,1) - midx) + 1);
DirichletBoundaryValue =  zeros(size(x,1),1);
end

%%
function Stress = myg(x,t)
Stress = zeros(size(x,1),1);
end

%%
function M = stima3(vertices)
d = size(vertices,2);
G = [ones(1,d+1);vertices'] \ [zeros(1,d);eye(d)];
M = det([ones(1,d+1);vertices']) * G * G' / prod(1:d);
end

%%
function heatsource = myf(x,t)
heatsource = ones(size(x,1),1);
end