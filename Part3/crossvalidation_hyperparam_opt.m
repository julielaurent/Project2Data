close all;
clear all;
clc;

load('dataset_ERP.mat');


%% k-fold cross-validation with Fischer

% Partition: K = 10
cp_labels = cvpartition (labels,'kfold',10);

% Initialization of error vector
N = 60;
errClassDiagLinTest = zeros(N, cp_labels.NumTestSets);
errClassDiagLinTrain = zeros(N,cp_labels.NumTestSets);

% Cross-validation
for i = 1:cp_labels.NumTestSets
    
    % Initialisation
    features_model = [];
    
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx = cp_labels.training(i);
    trainLabels = labels(trainIdx);
    testIdx = cp_labels.test(i);
    testLabels = labels(testIdx);
    
    % Rank of features on training set
    [orderedInd, orderedPower] = rankfeat(features(trainIdx,:),labels(trainIdx),'fisher');
        
   for j = 1:N
        features_model = [features_model, features(:,orderedInd(j))];

        trainSet = features_model(trainIdx,:);
        testSet = features_model(testIdx,:);
         
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

% Best number of features with min test error
meanTesterror = mean(errClassDiagLinTest,2);
minerror = min(meanTesterror);
nbfeature_minTesterror = find(meanTesterror == minerror);
nbfeature_minTesterror = nbfeature_minTesterror(1) % If several min value, select the first one

% Plot of the error for each cross-validation
figure('Color','w');
title('Training and Testing Error for each Number of Feature (Fischer Method)');
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

%% k-fold cross-validation with Correlation of Pearson

% Partition: K = 10
cp_labels = cvpartition (labels,'kfold',10);

% Initialization of error vector
N = 60;
errClassDiagLinTest = zeros(N, cp_labels.NumTestSets);
errClassDiagLinTrain = zeros(N,cp_labels.NumTestSets);

% Cross-validation
for i = 1:cp_labels.NumTestSets
    
    % Initialisation
    features_model = [];
    
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx = cp_labels.training(i);
    trainLabels = labels(trainIdx);
    testIdx = cp_labels.test(i);
    testLabels = labels(testIdx);
    
    % Rank of features on training set
    [orderedInd, orderedPower] = rankfeat(features(trainIdx,:),labels(trainIdx),'corr');
        
   for j = 1:N
        features_model = [features_model, features(:,orderedInd(j))];

        trainSet = features_model(trainIdx,:);
        testSet = features_model(testIdx,:);
         
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

% Best number of features
meanTesterror = mean(errClassDiagLinTest,2);
minerror = min(meanTesterror);
nbfeature_minTesterror = find(meanTesterror == minerror);
nbfeature_minTesterror = nbfeature_minTesterror(1) % If several min value, select the first one

% Plot of the error for each cross-validation
figure('Color','w');
title('Training and Testing Error for each Number of Feature (Pearson Correlation)');
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


%% Random classifier

% k-fold partition of our data
cp_labels = cvpartition (labels,'kfold',10);

% Initialization 
R = 1000; % Number of repetitions
N = 60; % Number of features max
errClassRandomTest = [];
meanTesterror = [];

for x = 1:R
    % Cross-validation
    for i = 1:cp_labels.NumTestSets
      
        % Initialization
        features_model = [];
    
         % Attention,ici le cp_N.taining rend les INDICES des train samples
         % Quand trainIdx = 1 -> sample qui va dans le trainSet
         trainIdx = cp_labels.training(i);
         trainLabels = labels(trainIdx);
         testIdx = cp_labels.test(i);
         testLabels = labels(testIdx);
        
         % Rank of features on training set: v?rifier si on laisse fisher
         [orderedInd, orderedPower] = rankfeat(features(trainIdx,:),labels(trainIdx),'fisher');
        
        for j = 1:N
             features_model = [features_model features(:,orderedInd(j))];   

             % Classifier construction --> vector 648X1 of random labels (0 or 1)
             Randomlabel = round(rand([648 1]));
                      
             % Attribute our random label to test and train set
             RandomlabelTest = Randomlabel(testIdx);
             RandomlabelTrain = Randomlabel(trainIdx);
        
             % Create our train and test set of inner loop for this model
             trainSet = features_model(trainIdx,:);
             testSet = features_model(testIdx,:);
        
             % Calculus of class error on test set -> testing error (NxK)
             errClassRandomTest(j,i) = classerror(testLabels, RandomlabelTest);
        end
    end

    % Mean
    meanTesterror_in = mean(errClassRandomTest,2); % Mean across folds
    meanTesterror(x) = mean(meanTesterror_in); % Mean for each repetition
end

minerror = min(meanTesterror); % Min test error across repetitions

% Plot of the mean error across folds for the 1 repetition
figure('Color','w');
title('Testing mean Error across folds for one repetition');
hold on;
plot(meanTesterror_in,'b','Linewidth',2);
legend('Mean testing error');
xlabel('Number of features');
ylabel('Class Error');
box off;
hold off;

% Plot of the mean error across folds for the 1000 repetitions
figure('Color','w');
title('Mean Test Error for each repetition');
hold on;
plot(meanTesterror,'b','Linewidth',2);
legend('Mean testing error');
xlabel('Repetition');
ylabel('Class Error');
box off;
hold off;