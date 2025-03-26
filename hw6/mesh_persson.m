figure(1)
fd = inline('ddiff(drectangle(p,-1,1,-1,1),drectangle(p,0,1,0,1))','p');
pfix = [-1,-1;-1,1;1,-1;1,0;0,1;0,0];
[p,t]=distmesh2d(fd,@huniform,0.15,[-1,-1;1,1],pfix);

figure(2)
phi = (0:5)'/5*2*pi+pi/10;
pfix1 = [cos(phi),sin(phi)];
r = sin(pi/5)/sind(108/2)*cosd(72)/cosd(108/2);
phi = (0:5)'/5*2*pi-pi/10;
pfix2 = r*[cos(phi),sin(phi)];
fd = @(p) ddiff(dpoly(p,pfix1),dpoly(p,pfix2));
[p,t]=distmesh2d(fd,@huniform,0.15,[-1,-1;1,1],[pfix1;pfix2]);

figure(3)
fd = @(p) ddiff(max(sqrt(sum(p.^2,2))-1,p(:,2)),sqrt((p(:,1)+0.5).^2+(p(:,2)+0.4).^2)-0.25);
fd = @(p) ddiff(fd(p),sqrt((p(:,1)-0.5).^2+(p(:,2)+0.4).^2)-0.25);
[p,t]=distmesh2d(fd,@huniform,0.08,[-1,-1;1,1],[-1,0;1,0]);
