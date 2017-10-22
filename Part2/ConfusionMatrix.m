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

% J'ai utilis? diagquadr parce que ? priori le meilleur?

cp_N = cvpartition(N,'kfold',10); % Outputs: Trainsize and Testsize contain the size (= number of samples) of each train/test set

% Initialization of error vectors
errClassDiagQuadr = zeros(cp_N.NumTestSets,1);

% For a different testSet i each time
for i = 1:cp_N.NumTestSets
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx = cp_N.training(i);
    trainSet = features(trainIdx,:);
    trainLabels = labels(trainIdx);
   
    % Attention, ici le cp_N.test rend les INDICES des test samples
    % Quand testIdx = 1 -> sample va dans le testSet
    testIdx = cp_N.test(i);
    testSet = features(testIdx,:);
    testLabels = labels(testIdx);
    
    % Calculus of class errors
    DiagQuadrclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagQuadratic');
    DiagQuadr_y = predict(DiagQuadrclassifier,testSet);
    errClassDiagQuadr(i) = classerrorOriginal(testLabels, DiagQuadr_y);
end




predicted = ...
CM = confusionmat(labels,predicted);
