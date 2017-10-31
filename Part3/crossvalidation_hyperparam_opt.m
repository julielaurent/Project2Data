close all;
clear all;
clc;

load('dataset_ERP.mat');

%% Rank of features
[orderedInd, orderedPower] = rankfeat(features,labels,'fisher');

%% Normal k-fold cross-validation

cp_labels = cvpartition (labels,'kfold',10);

% Initialization of error vector
N = 5;
errClassDiagLinTest = zeros(N, cp_labels.NumTestSets);
errClassDiagLinTrain = zeros(N,cp_labels.NumTestSets);
features_model = [];

for j = 1:N
    features_model = [features_model, features(:,orderedInd(j))];
    % For a different testSet i each time
    for i = 1:cp_labels.NumTestSets

        % Attention,ici le cp_N.taining rend les INDICES des train samples
        % Quand trainIdx = 1 -> sample qui va dans le trainSet
        trainIdx = cp_labels.training(i);
        trainSet = features_model(trainIdx,:);
        trainLabels = labels(trainIdx);

        % Attention, ici le cp_N.test rend les INDICES des test samples
        % Quand testIdx = 1 -> sample va dans le testSet
        testIdx = cp_labels.test(i);
        testSet = features_model(testIdx,:);
        testLabels = labels(testIdx);
        
        % Classifier construction
        DiagLinclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagLinear');

        % Calculus of class error on test set -> testing error (NxK)
        DiagLin_yTest = predict(DiagLinclassifier,testSet);
        errClassDiagLinTest(j,i) = classerror(testLabels, DiagLin_yTest);
        
        % Calculus of class error on train set -> training error (NxK)
        DiagLin_yTrain = predict(DiagLinclassifier,trainSet);
        errClassDiagLinTrain(j,i) = classerror(trainLabels, DiagLin_yTrain);
    end
end    

% % Mean error and std accross folds (one for each model)
% cvErrDiagLinTest = zeros(N,1);
% cvErrDiagLinTrain = zeros(N,1);
% cvStdDiagLinTest = zeros(N,1);
% cvStdDiagLinTrain = zeros(N,1);
% 
% for j = 1:N
%     % Mean of errors -> matrix NX1
%     cvErrDiagLinTest(j,1) = mean(errClassDiagLinTest(j,:));
%     cvErrDiagLinTrain(j,1) = mean(errClassDiagLinTrain(j,:));
%     
%     % Standard deviation of error -> matrix NX1
%     cvStdDiagLinTest(j,1) = std(errClassDiagLinTest(j,:));
%     cvStdDiagLinTrain(j,1) = std(errClassDiagLinTrain(j,:));
% end

% Plot of the error for each cross-validation
figure('Color','w');
title('Training and Testing Error for each Number of Feature');
hold on;
plot(errClassDiagLinTest,'b');
hold on;
p1 = plot(mean(errClassDiagLinTest,2),'b','Linewidth',2);
legend('Mean testing error');
hold on;
plot(errClassDiagLinTrain,'r');
hold on;
p2 = plot(mean(errClassDiagLinTrain,2),'r','Linewidth',2);
legend([p1, p2],'Mean testing error', 'Mean training error');
xlabel('Number of features');
ylabel('Class Error');
xticks([1 2 3 4 5]);
box off;
hold off;