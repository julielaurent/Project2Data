close all;
clear all;
clc;

load('dataset_ERP.mat');


%% k-fold cross-validation with Fischer

% Rank of features
[orderedInd, orderedPower] = rankfeat(features,labels,'fisher');

cp_labels = cvpartition (labels,'kfold',10);

% Initialization of error vector
N = 60;
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

% Best number of features: 15
meanTesterror = mean(errClassDiagLinTest,2);
minerror = min(meanTesterror);
nbfeature_minTesterror = find(meanTesterror == minerror);

% Plot of the error for each cross-validation
figure('Color','w');
title('Training and Testing Error for each Number of Feature');
hold on;
plot(errClassDiagLinTest,'b--','Linewidth',0.01);
hold on;
p1 = plot(mean(errClassDiagLinTest,2),'b','Linewidth',2);
legend('Mean testing error');
hold on;
plot(errClassDiagLinTrain,'r--','Linewidth',0.01);
hold on;
p2 = plot(mean(errClassDiagLinTrain,2),'r','Linewidth',2);
legend([p1, p2],'Mean testing error', 'Mean training error');
xlabel('Number of features');
ylabel('Class Error');
xticks(0:5:60);
box off;
hold off;
% Q?Alice: on voit que un type d'erreur la non? (? part les means)

%% k-fold cross-validation with Correlation of Pearson

% Rank of features
[orderedInd, orderedPower] = rankfeat(features,labels,'corr');

cp_labels = cvpartition (labels,'kfold',10);

% Initialization of error vector
N = 60;
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

% Best number of features: 15 also
meanTesterror = mean(errClassDiagLinTest,2);
minerror = min(meanTesterror);
nbfeature_minTesterror = find(meanTesterror == minerror);

% Plot of the error for each cross-validation
figure('Color','w');
title('Training and Testing Error for each Number of Feature');
hold on;
plot(errClassDiagLinTest,'b--','Linewidth',0.01);
hold on;
p1 = plot(mean(errClassDiagLinTest,2),'b','Linewidth',2);
legend('Mean testing error');
hold on;
plot(errClassDiagLinTrain,'r--','Linewidth',0.01);
hold on;
p2 = plot(mean(errClassDiagLinTrain,2),'r','Linewidth',2);
legend([p1, p2],'Mean testing error', 'Mean training error');
xlabel('Number of features');
ylabel('Class Error');
xticks(0:5:60);
box off;
hold off;
% Q?Alice: on voit que un type d'erreur la non? (? part les means)

%% Random classifier

% Rank of features
[orderedInd, orderedPower] = rankfeat(features,labels,'fisher');

% k-fold partition of our data
cp_labels = cvpartition (labels,'kfold',10);

% Classifier construction --> vector 648X1 of random labels (0 or 1)
Randomlabel = round(rand([648 1]));


% Choice of the 15 best features
features_model = [];
for n = 1:15
    features_model = [features_model features(:,orderedInd(n))];   
end
% features_model = [features(:,orderedInd(1)), features(:,orderedInd(2))];


% Initialization of error vector
N = 1000;
errClassRandomTest = [];

for j = 1:N
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



