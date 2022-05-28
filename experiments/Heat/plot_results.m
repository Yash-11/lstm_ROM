%%
clear all; close all; clc
addpath(genpath('../../src/Heat/data'))
addpath(genpath('./firstTry'))


% load('pod_basis.mat')
load('Gcoor.mat')
load('matConnec.mat')


pred = h5read('predDataTest_epoch15000_.hdf5','/pred');
target = h5read('predDataTest_epoch15000_.hdf5','/target');

% Pred = U*pred;
% Target = U*target;


faces = matConnec(1:64, :);

%%
Resultat(Gcoor, pred(:, 20), faces);
Resultat(Gcoor, target(:, 20), faces);

function []= Resultat(xy,T,cn)

 xy= xy*5;
% T=sol;
% ne=nelt;
% nn=nnt;
% cn=Connec(1:nelt,:);

% affichage graphique couleur
figure
% title('solution en température')
hold on
patch('Vertices',xy,'Faces',cn,'FaceVertexCData',T,'FaceColor','interp')    % interpolation
colorbar ('South')              % south ???

% affichage numerique des noeuds
for i=1:size(xy,1)
    plot(xy(i,1),xy(i,2),'.k')
    if xy(i,2)>=xy(end,2)/2
        si=text(xy(i,1),xy(i,2),[' ',num2str(T(i),4),'°C'],'VerticalAlignment','bottom');
    else
        si=text(xy(i,1),xy(i,2),[' ',num2str(T(i),4),'°C'],'VerticalAlignment','top');
    end
        set(si,'Color','k','FontSize',10)
end
xlabel('x (m)'), ylabel('y (m)'),axis('equal')
hold off
end
