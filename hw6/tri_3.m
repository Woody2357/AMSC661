N = 100;
node = [];
edge = [];
for n=1:N
    node = [node; cos(n/N*pi),sin(-n/N*pi)];
    edge = [edge; n,mod(n,N)+1];
end
for n=1:N
    node = [node; -0.5+0.2*cos(n/N*2*pi),-0.4+0.2*sin(n/N*2*pi)];
    edge = [edge; N+n,mod(n,N)+N+1];
end
for n=1:N
    node = [node;0.5+0.2*cos(n/N*2*pi),-0.4+0.2*sin(n/N*2*pi)];
    edge = [edge;2*N+n,mod(n,N)+2*N+1];
end

%------------------------------------------- call mesh-gen.
[vert,etri, ...
tria,tnum] = refine2(node,edge) ;

%------------------------------------------- draw tria-mesh
figure;
patch('faces',tria(:,1:3),'vertices',vert, ...
    'facecolor','w', ...
    'edgecolor',[.2,.2,.2]) ;
hold on; axis image off;
patch('faces',edge(:,1:2),'vertices',node, ...
    'facecolor','w', ...
    'edgecolor',[.1,.1,.1], ...
    'linewidth',1.5) ;

%------------------------------------------- call mesh-gen.
hfun = +.5 ;            % uniform "target" edge-lengths

[vert,etri, ...
tria,tnum] = refine2(node,edge,[],[],hfun) ;

%------------------------------------------- draw tria-mesh
figure;
patch('faces',tria(:,1:3),'vertices',vert, ...
    'facecolor','w', ...
    'edgecolor',[.2,.2,.2]) ;
hold on; axis image off;
patch('faces',edge(:,1:2),'vertices',node, ...
    'facecolor','w', ...
    'edgecolor',[.1,.1,.1], ...
    'linewidth',1.5) ;

drawnow;

set(figure(1),'units','normalized', ...
    'position',[.05,.50,.30,.35]) ;
set(figure(2),'units','normalized', ...
    'position',[.35,.50,.30,.35]) ;