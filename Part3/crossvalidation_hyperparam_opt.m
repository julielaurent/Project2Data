close all;
clear all;
clc;

load('dataset_ERP.mat');


% %% k-fold cross-validation with Fischer
% 
% % Rank of features
% [orderedInd, orderedPower] = rankfeat(features,labels,'fisher');
% 
% cp_labels = cvpartition (labels,'kfold',10);
% 
% % Initialization of error vector
% N = 10;
% errClassDiagLinTest = zeros(N, cp_labels.NumTestSets);
% errClassDiagLinTrain = zeros(N,cp_labels.NumTestSets);
% features_model = [];
% 
% for j = 1:N
%     features_model = [features_model, features(:,orderedInd(j))];
%     % For a different testSet i each time
%     for i = 1:cp_labels.NumTestSets
% 
%         % Attention,ici le cp_N.taining rend les INDICES des train samples
%         % Quand trainIdx = 1 -> sample qui va dans le trainSet
%         trainIdx = cp_labels.training(i);
%         trainSet = features_model(trainIdx,:);
%         trainLabels = labels(trainIdx);
% 
%         % Attention, ici le cp_N.test rend les INDICES des test samples
%         % Quand testIdx = 1 -> sample va dans le testSet
%         testIdx = cp_labels.test(i);
%         testSet = features_model(testIdx,:);
%         testLabels = labels(testIdx);
%         
%         % Classifier construction
%         DiagLinclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagLinear');
% 
%         % Calculus of class error on test set -> testing error (NxK)
%         DiagLin_yTest = predict(DiagLinclassifier,testSet);
%         errClassDiagLinTest(j,i) = classerror(testLabels, DiagLin_yTest);
%         
%         % Calculus of class error on train set -> training error (NxK)
%         DiagLin_yTrain = predict(DiagLinclassifier,trainSet);
%         errClassDiagLinTrain(j,i) = classerror(trainLabels, DiagLin_yTrain);
%     end
% end    
% 
% % Plot of the error for each cross-validation
% figure('Color','w');
% title('Training and Testing Error for each Number of Feature');
% hold on;
% plot(errClassDiagLinTest,'b--','Linewidth',0.01);
% hold on;
% p1 = plot(mean(errClassDiagLinTest,2),'b','Linewidth',2);
% legend('Mean testing error');
% hold on;
% plot(errClassDiagLinTrain,'r--','Linewidth',0.01);
% hold on;
% p2 = plot(mean(errClassDiagLinTrain,2),'r','Linewidth',2);
% legend([p1, p2],'Mean testing error', 'Mean training error');
% xlabel('Number of features');
% ylabel('Class Error');
% xticks([1 2 3 4 5 6 7 8 9 10]);
% box off;
% hold off;
% 
% %% k-fold cross-validation with Correlation of Pearson
% 
% % Rank of features
% [orderedInd, orderedPower] = rankfeat(features,labels,'corr');
% 
% cp_labels = cvpartition (labels,'kfold',10);
% 
% % Initialization of error vector
% N = 10;
% errClassDiagLinTest = zeros(N, cp_labels.NumTestSets);
% errClassDiagLinTrain = zeros(N,cp_labels.NumTestSets);
% features_model = [];
% 
% for j = 1:N
%     features_model = [features_model, features(:,orderedInd(j))];
%     % For a different testSet i each time
%     for i = 1:cp_labels.NumTestSets
% 
%         % Attention,ici le cp_N.taining rend les INDICES des train samples
%         % Quand trainIdx = 1 -> sample qui va dans le trainSet
%         trainIdx = cp_labels.training(i);
%         trainSet = features_model(trainIdx,:);
%         trainLabels = labels(trainIdx);
% 
%         % Attention, ici le cp_N.test rend les INDICES des test samples
%         % Quand testIdx = 1 -> sample va dans le testSet
%         testIdx = cp_labels.test(i);
%         testSet = features_model(testIdx,:);
%         testLabels = labels(testIdx);
%         
%         % Classifier construction
%         DiagLinclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagLinear');
% 
%         % Calculus of class error on test set -> testing error (NxK)
%         DiagLin_yTest = predict(DiagLinclassifier,testSet);
%         errClassDiagLinTest(j,i) = classerror(testLabels, DiagLin_yTest);
%         
%         % Calculus of class error on train set -> training error (NxK)
%         DiagLin_yTrain = predict(DiagLinclassifier,trainSet);
%         errClassDiagLinTrain(j,i) = classerror(trainLabels, DiagLin_yTrain);
%     end
% end    
% 
% % Plot of the error for each cross-validation
% figure('Color','w');
% title('Training and Testing Error for each Number of Feature');
% hold on;
% plot(errClassDiagLinTest,'b--','Linewidth',0.01);
% hold on;
% p1 = plot(mean(errClassDiagLinTest,2),'b','Linewidth',2);
% legend('Mean testing error');
% hold on;
% plot(errClassDiagLinTrain,'r--','Linewidth',0.01);
% hold on;
% p2 = plot(mean(errClassDiagLinTrain,2),'r','Linewidth',2);
% legend([p1, p2],'Mean testing error', 'Mean training error');
% xlabel('Number of features');
% ylabel('Class Error');
% xticks([1 2 3 4 5 6 7 8 9 10]);
% box off;
% hold off;

%% Random classifier

% Rank of features
[orderedInd, orderedPower] = rankfeat(features,labels,'fisher');

% k-fold partition of our data
cp_labels = cvpartition (labels,'kfold',10);

% Classifier construction --> vector 648X1 of random labels (0 or 1)
Randomlabel = round(rand([648 1]));

% Initialization of error vector
N = 1000;
%errClassRandomTest = zeros(N, cp_labels.NumTestSets);
%errClassRandomTrain = zeros(N,cp_labels.NumTestSets);
errClassRandomTest = [];
features_model = [features(:,orderedInd(1)), features(:,orderedInd(2))];

for j = 1:N
    %features_model = [features_model, features(:,orderedInd(2))];
    
    % Classifier construction --> vector 648X1 of random labels (0 or 1)
     Randomlabel = round(rand([648 1]));
    
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
                      
        % Attribute our random label to test and train set
        RandomlabelTest = Randomlabel(testIdx);
        RandomlabelTrain = Randomlabel(trainIdx);
        
        % Calculus of class error on test set -> testing error (NxK)
        errClassRandomTest(j,i) = classerror(testLabels, RandomlabelTest);
    end
    cv_errClassRandomTest = mean(errClassRandomTest,2);
end

min_cv_errClassRandomTest = min(cv_errClassRandomTest)
find(cv_errClassRandomTest == min_cv_errClassRandomTest)

% Plot of the mean error across folds for the 1000 repetitions
figure('Color','w');
title('Testing mean Error across folds for each repetition');
hold on;
plot(cv_errClassRandomTest,'b','Linewidth',2);
legend('Mean testing error');
xlabel('Repetition');
ylabel('Class Error');
box off;
hold off;



