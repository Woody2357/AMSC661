function [A,b] = MyFEMCat()
close all
bdry = [0,0;0,3;3,0;3,3];
fd = @(p) drectangle(p,0,3,0,3);
N = round(2*pi/0.1);
phi = (0:N)'/N*2*pi;
pfix = [1.5+cos(phi),1.5+sin(phi)];
pfix = [bdry;pfix];

[pts,tri] = distmesh2d(fd,@huniform,0.1,[0,0;3,3],pfix);
cond = conductivity(tri,pts);
eps = 1e-4;
dirichlet1 = find(abs(pts(:,1))<eps);
dirichlet2 = find(abs(pts(:,1)-3)<eps);
dirichlet = [dirichlet1;dirichlet2];

Npts = size(pts,1);
FreeNodes = setdiff(1:Npts,dirichlet);
A = sparse(Npts,Npts);
b = sparse(Npts,1);

Ntri = size(tri,1);
for j = 1:Ntri
    A(tri(j,:),tri(j,:)) = A(tri(j,:),tri(j,:))+cond(j)*stima3(pts(tri(j,:),:));
end
for j = 1:Ntri
    b(tri(j,:))=0;
end

u = sparse(size(pts,1),1);
u(dirichlet1) = 0;
u(dirichlet2) = 1;
b = b - A*u;
u(FreeNodes) = A(FreeNodes,FreeNodes)\b(FreeNodes);

figure(1)
trisurf(tri,pts(:,1),pts(:,2),full(u)','facecolor','interp')
hold on
axis ij
view(2)

abs_current_centors = zeros(Ntri,1);
for j = 1:Ntri
    vertices = pts(tri(j,:),:);
    d = size(vertices,2);
    G = [ones(1,d+1);vertices'] \ [zeros(1,d);eye(d)];
    I = -cond(j)*G'*u(tri(j,:));
    abs_current_centers(j) = norm(I);
end

abs_current_verts = zeros(Npts,1);
count_tri = zeros(Npts,1);
for j = 1:Ntri
    abs_current_verts(tri(j,:)) = abs_current_verts(tri(j,:))+abs_current_centers(j);
    count_tri(tri(j,:)) = count_tri(tri(j,:)) + 1;
end
abs_current_verts = abs_current_verts./count_tri;

figure(2);
trisurf(tri,pts(:,1),pts(:,2),full(abs_current_verts)', 'facecolor','interp')
hold on
axis ij
view(2)
end

function Stress = myg(x)
    Stress = zeros(size(x,1),1);
end

function M = stima3(vertices)
    d = size(vertices,2);
    G = [ones(1,d+1);vertices'] \ [zeros(1,d);eye(d)];
    M = det([ones(1,d+1);vertices']) * G * G' / prod(1:d);
end

function cond = conductivity(tri,pts)
    a1 = 1.2;
    a2 = 1;
    center = (pts(tri(:,1),:) + pts(tri(:,2),:) + pts(tri(:,3),:))/3;
    dist = sqrt((center(:,1) - 1.5).^2 + (center(:,2) - 1.5).^2);
    cond = a1*(dist<=1) + a2*(dist>1);
end

% mesh_flag = 1; % generate mesh
% c = imread('cat.png');
% cc = sum(c,3);
% h = contour(cc,[1 1]);
% % extract contours of face and eyes
% ind = find(h(1,:)==1);
% c1 = h(:,2:6:ind(2)-1)';  % face
% c2 = h(:,ind(3)+1:6:ind(4)-1)'; % eye
% c3 = h(:,ind(4)+1:6:ind(5)-1)'; % another eye
% lc1 = length(c1); 
% lc2 = length(c2);
% lc3 = length(c3);
% % connect boundary points
% a1 = [1 : lc1]';
% e1 = [a1, circshift(a1,[-1 0])];
% a2 = [1 : lc2]';
% e2 = [lc1 + a2, lc1 + circshift(a2,[-1 0])];
% a3 = [1 : lc3]';
% e3 = [lc1 + lc2 + a3, lc1 + lc2 + circshift(a3,[-1 0])];
% bdry = [c1; c2; c3]; % coordinates of boundary points
% bdry_connect = [e1; e2; e3]; % indices of endpoints of boundary segments
% if mesh_flag == 0    
%     % make mesh
%     [pts,~,tri,~] = refine2(bdry,bdry_connect);
%     save('MyFEMcat_mesh.mat','pts','tri');
% else
%     msh = load('MyFEMcat_mesh.mat');
%     pts = msh.pts;
%     tri = msh.tri;
% end
% %---------------------------------------------- do mesh-opt.
% 
% % pts is a N-by-2 array with coordinated of the mesh points
% % tri is a Ntriag-by-3 array of indices of the triangular elements
% 
% %%
% % Find boundary points with homogeneous Neumann BCs
% % The number of rows in neumann is the number of boundary intervals with
% % Neumann BCs
% 
% % Each row of neumann contains indices of endpoints of the corresponding
% % boundary interval
% ind = zeros(lc1,1);
% for i = 1 : lc1
%     ind(i) = find(pts(:,1) == c1(i,1) & pts(:,2) == c1(i,2));
% end
% neumann = [ind,circshift(ind,[-1, 0])];
% 
% % Find boundary points with Dirichlet BCs
% % dirichlet is a column vector of indices of points with Dirichlet BCs
% dirichlet = zeros(lc2 + lc3,1);
% ind = zeros(lc2,1);
% for i = 1 : lc2
%     dirichlet(i) = find(pts(:,1) == c2(i,1) & pts(:,2) == c2(i,2));
% end
% ind = zeros(lc3,1);
% for i = 1 : lc3
%     dirichlet(lc2 + i) = find(pts(:,1) == c3(i,1) & pts(:,2) == c3(i,2));
% end
% 
% % call FEM
% u = FEM2D(pts,tri,neumann,dirichlet);
% 
% % graphic representation
% figure;
% trisurf(tri,pts(:,1),pts(:,2),full(u),'facecolor','interp')
% hold on
% axis ij
% view(2)
% xlabel('x','Fontsize',20);
% ylabel('y','Fontsize',20);
% set(gca,'Fontsize',20);
% colorbar
% end
% 
% %% FEM
% function u = FEM2D(pts,tri,neumann,dirichlet)
% Npts = size(pts,1);
% Ntri = size(tri,1);
% FreeNodes = setdiff(1:Npts,dirichlet); %mesh points with unknown values of u
% A = sparse(Npts,Npts);
% b = sparse(Npts,1);
% 
% %% Assembly
% %% The Stiffness matrix
% for j = 1:Ntri % for all triangles    
%   A(tri(j,:),tri(j,:)) = A(tri(j,:),tri(j,:)) + stima3(pts(tri(j,:),:));
%   % stima3_2 computes M = 0.5*|T_j|*G*G';
% end
% 
% %% The Right-hand side, i.e., the load vector
% % Volume Forces
% for j = 1:Ntri
%   b(tri(j,:)) = 0;  % for the case where f = 0
% end
% 
% % Neumann conditions
% for j = 1 : size(neumann,1)
%   b(neumann(j,:)) = b(neumann(j,:)) + norm(pts(neumann(j,1),:)- ...
%       pts(neumann(j,2),:)) * myg(sum(pts(neumann(j,:),:))/2)/2;
% end
% 
% % Dirichlet conditions 
% u = sparse(Npts,1);
% u(dirichlet) = myu_d(pts(dirichlet,:));
% b = b - A * u;
% 
% % Computation of the solution
% u(FreeNodes) = A(FreeNodes,FreeNodes) \ b(FreeNodes);
% 
% end
% 
% %%
% function DirichletBoundaryValue = myu_d(x)
% xmin = min(x(:,1));
% xmax = max(x(:,1));
% midx = 0.5*(xmin + xmax);
% DirichletBoundaryValue =  0.5 * (sign(x(:,1) - midx) + 1);
% end
% 
% %%
% function Stress = myg(x)
% Stress = zeros(size(x,1),1);
% end
% 
% %
% function M = stima3(verts)
% G = [ones(1,3);verts'] \ [zeros(1,2);eye(2)];
% M = 0.5*det([ones(1,3);verts']) * G * G';
% end