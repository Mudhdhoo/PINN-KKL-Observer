clear; clc; close all;

pwd;
currentFolder = pwd;

%%
x_truth = load('revduff_truth.mat');% load('vdp_truth.mat');
x_noise = load('revduff_noise.mat');%load('vdp_noise.mat')
x_est = load('revduff_est.mat');%load('vdp_est.mat')

figure; 
% rectangle('Position',[-0.25 -1 2 2],'FaceColor','#808080','EdgeColor','k',...
%     'LineWidth',1)
% hold on;
plot(x_truth.states(:,1), x_truth.states(:,2),'r.','LineWidth',2)
hold on;
plot(x_est.states(:,1), x_est.states(:,2),'k','LineWidth',2)

xlabel('$x_1$')
ylabel('$x_2$')

set(gcf,'Renderer','painters');
figureHandle = gcf;
set(findall(figureHandle,'type','text'),'fontSize',20,'interpreter','latex');
set(findall(figureHandle,'type','axes'),'Color','w','fontSize',20);
set(findall(figureHandle,'type','axes'),'TickLabelInterpreter','latex');
axis tight
savePDF1(figureHandle.Position([3]),figureHandle.Position([4]))

