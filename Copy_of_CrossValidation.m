close all;
clear all;
clc;

load('dataset_ERP.mat');

features = features(:,1:5:300);
size_labels = size(labels);

% Faut-il randomizer les data ? -> remplacer "features" par "newFeatures"
% Possible réponse à ma question : non parce que cv partition le fait deja
% diféremment à chaque fois
%randFeatures = randperm(648);
%newFeatures = features(randFeatures,:);

%% k-fold cross-validation
% Split data into k subsets of equal size
% Use k-1 subsets to train the classifier
% Remaining subset will be used for the testing --> for that make an
% iteration in order to one time use the 1 testing, and 2,3,4.. training,
% then the 2 testing, and 1,3,4,5... training etc...

N = size_labels(1); % N = total number of samples
cp_N = cvpartition(N,'kfold',10); % Outputs: Trainsize and Testsize contain the size (= number of samples) of each train/test set
cp_labels = cvpartition (labels,'kfold',10);
C = cvpartition(N,'LeaveOut'); %"leave-one-out" cross validation

% Initialization of error vectors
errClassLin = zeros(cp_N.NumTestSets,1);
errClassDiagLin = zeros(cp_N.NumTestSets,1);
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
    testIdx= cp_N.test(i);
    testSet = features(testIdx,:);
    testLabels = labels(testIdx);
    
    % Calculus of class errors
    Linclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'linear');
    Lin_y = predict(Linclassifier,testSet);
    errClassLin(i) = classerrorOriginal(testLabels, Lin_y);
    
    DiagLinclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagLinear');
    DiagLin_y = predict(DiagLinclassifier,testSet);
    errClassDiagLin(i) = classerrorOriginal(testLabels, DiagLin_y);
    
    DiagQuadrclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagQuadratic');
    DiagQuadr_y = predict(DiagQuadrclassifier,testSet);
    errClassDiagQuadr(i) = classerrorOriginal(testLabels, DiagQuadr_y);
end

%% Mean of errors
cvErrLin = sum(errClassLin)/sum(cp_N.TestSize);
cvErrDiagLin = sum(errClassDiagLin)/sum(cp_N.TestSize);
cvErrDiagQuadr = sum(errClassDiagQuadr)/sum(cp_N.TestSize);

%% Standard deviations of errors
cvStdLin = std(errClassLin);
cvStdDiagLin = std(errClassDiagLin);
cvStdDiagQuadr = std(errClassDiagQuadr);

%% Plot
figure('Color','w');
title('Cross Validation Error');
hold on;
barwitherr([cvStdDiagLin ; cvStdLin ; cvStdDiagQuadr], [cvErrDiagLin ; cvErrLin ; cvErrDiagQuadr]);
ylabel('Class Error');
xticks([1 2 3]);
xticklabels({'DiagLinear','Linear','DiagQuadratic'});
axis([0.5 3.5 0 0.1]);
box off;
box off;
hold off;