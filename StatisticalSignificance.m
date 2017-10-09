close all;
clear all;
clc;

load('dataset_ERP.mat');

%Class A: label '1'
%Class B: label '0'

x = features(find(labels == 1), :); %vector of class A
y = features(find(labels == 0), :); %vector of class B


for k=300:400
    figure('Name',num2str(k),'color','w');
%     title(['Sample' num2str(k)])
    histogram(x(:,k));
    hold on
    histogram(y(:,k));
    hold off
    j = waitforbuttonpress;
end


%Confidence Interval
%boxplot(features(:,301),'Notch','on');


%t-test

% 
% [h,p]=ttest2(x,y);