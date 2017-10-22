close all;
clear all;
clc;

load('dataset_ERP.mat');

features = features(:,1:5:300);
labels = labels(:,1:5:300);
%size_labels = size(labels);


%% Confusion matrix
%  CM = confusionmat(G,GHAT) returns the confusion matrix CM determined by 
%     the known group labels G and the predicted group labels GHAT. G and
%     GHAT are grouping variables with the same number of observations.


predicted = ...
CM = confusionmat(labels,predicted);