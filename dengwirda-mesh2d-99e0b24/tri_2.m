theta = exp(i*2*pi/5);

node = [                % list of xy "node" coordinates
    real(i), imag(i)                % outer square
    real(theta*i), imag(theta*i)
    real(theta^2*i), imag(theta^2*i)
    real(theta^3*i), imag(theta^3*i)
    real(theta^4*i), imag(theta^4*i)
    real(-i/3), imag(-i/3)  % inner square
    real(theta*-i/3), imag(theta*-i/3)
    real(theta^2*-i/3), imag(theta^2*-i/3)
    real(theta^3*-i/3), imag(theta^3*-i/3)
    real(theta^4*-i/3), imag(theta^4*-i/3)] ;

edge = [                % list of "edges" between nodes
    1, 2                % outer square
    2, 3
    3, 4
    4, 5
    5, 1                % inner square
    6, 7
    7, 8
    8, 9
    9, 10
    10, 6] ;

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