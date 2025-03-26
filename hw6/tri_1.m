node = [                % list of xy "node" coordinates
    -1, 1                % outer square
    0, 1
    0, 0
    1, 0
    1, -1                % inner square
    -1, -1 ] ;

edge = [                % list of "edges" between nodes
    1, 2                % outer square
    2, 3
    3, 4
    4, 5
    5, 6                % inner square
    6, 1 ] ;

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