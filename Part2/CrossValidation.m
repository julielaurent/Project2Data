close all;
clear all;
clc;

load('dataset_ERP.mat');

features = features(:,1:5:300);
size_labels = size(labels);


%% Normal k-fold cross-validation
% Split data into k subsets of equal size
% Use k-1 subsets to train the classifier
% Remaining subset will be used for the testing --> for that make an
% iteration in order to one time use the 1 testing, and 2,3,4.. training,
% then the 2 testing, and 1,3,4,5... training etc...

N = size_labels(1); % N = total number of samples
cp_N = cvpartition(N,'kfold',10); % Outputs: Trainsize and Testsize contain the size (= number of samples) of each train/test set
cp_labels = cvpartition (labels,'kfold',10);

% Initialization of error vectors
errClassLin = zeros(cp_labels.NumTestSets,1);
errClassDiagLin = zeros(cp_labels.NumTestSets,1);
errClassDiagQuadr = zeros(cp_labels.NumTestSets,1);
ratiosTrain = zeros(cp_labels.NumTestSets,1);
ratiosTest = zeros(cp_labels.NumTestSets,1);

% For a different testSet i each time
for i = 1:cp_labels.NumTestSets
    
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx = cp_labels.training(i);
    trainSet = features(trainIdx,:);
    trainLabels = labels(trainIdx);
   
    % Attention, ici le cp_N.test rend les INDICES des test samples
    % Quand testIdx = 1 -> sample va dans le testSet
    testIdx = cp_labels.test(i);
    testSet = features(testIdx,:);
    testLabels = labels(testIdx);
    
    % Ratios
    ratiosTrain(i) = size(find(trainLabels == 1))/size(find(trainLabels == 0));
    ratiosTest(i) = size(find(testLabels == 1))/size(find(testLabels == 0));
    
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

% % Mean of errors
cvErrLin = mean(errClassLin);
cvErrDiagLin = mean(errClassDiagLin);
cvErrDiagQuadr = mean(errClassDiagQuadr);

% Standard deviations of errors
cvStdLin = std(errClassLin);
cvStdDiagLin = std(errClassDiagLin);
cvStdDiagQuadr = std(errClassDiagQuadr);

% Mean and std of ratios
rMTrain = mean(ratiosTrain);
rMTest = mean(ratiosTest);
rSTDTrain = std(ratiosTrain);
rSTDTest = std(ratiosTest);


% Plot
figure('Color','w');
title('10-fold Cross Validation Error');
hold on;
barwitherr([cvStdDiagLin ; cvStdLin ; cvStdDiagQuadr], [cvErrDiagLin ; cvErrLin ; cvErrDiagQuadr]);
ylabel('Class Error');
xticks([1 2 3]);
xticklabels({'DiagLinear','Linear','DiagQuadratic'});
axis([0.5 3.5 0 0.7]);
box off;
hold off;

% Leave One Out Cross Validation

C = cvpartition(labels,'LeaveOut'); %"leave-one-out" cross validation
% Error Calculation
errClassLinC = zeros(C.NumTestSets,1);
errClassDiagLinC = zeros(C.NumTestSets,1);
errClassDiagQuadrC = zeros(C.NumTestSets,1);

for i = 1:C.NumTestSets
    
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdxC = C.training(i);
    trainSetC = features(trainIdxC,:);
    trainLabelsC = labels(trainIdxC);
    
    % Attention, ici le cp_N.test rend les INDICES des test samples
    % Quand testIdx = 1 -> sample va dans le testSet
    testIdxC = C.test(i);
    testSetC = features(testIdxC,:);
    testLabelsC = labels(testIdxC);
    
    % Calculus of class errors
    LinclassifierC = fitcdiscr(trainSetC,trainLabelsC,'discrimtype', 'linear');
    Lin_yC = predict(LinclassifierC,testSetC);
    errClassLinC(i) = classerrorOriginal(testLabelsC, Lin_yC);
    
    DiagLinclassifierC = fitcdiscr(trainSetC,trainLabelsC,'discrimtype', 'diagLinear');
    DiagLin_yC = predict(DiagLinclassifierC,testSetC);
    errClassDiagLinC(i) = classerrorOriginal(testLabelsC, DiagLin_yC);
    
    DiagQuadrclassifierC = fitcdiscr(trainSetC,trainLabelsC,'discrimtype', 'diagQuadratic');
    DiagQuadr_yC = predict(DiagQuadrclassifierC,testSetC);
    errClassDiagQuadrC(i) = classerrorOriginal(testLabelsC, DiagQuadr_yC);
    
end

% Mean of errors 
cvErrLinC = mean(errClassLinC);
cvErrDiagLinC = mean(errClassDiagLinC);
cvErrDiagQuadrC = mean(errClassDiagQuadrC);

% Standard deviations of errors
cvStdLinC = std(errClassLinC);
cvStdDiagLinC = std(errClassDiagLinC);
cvStdDiagQuadrC = std(errClassDiagQuadrC);

% Plot
figure('Color','w');
title('Leave One Out Cross Validation Error');
hold on;
barwitherr([cvStdDiagLinC ; cvStdLinC ; cvStdDiagQuadrC], [cvErrDiagLinC ; cvErrLinC ; cvErrDiagQuadrC]);
ylabel('Class Error');
xticks([1 2 3]);
xticklabels({'DiagLinear','Linear','DiagQuadratic'});
axis([0.5 3.5 -0.3 0.7]);
box off;
hold off;

%% Different repartitions

N = size_labels(1); % N = total number of samples
cp_labels = cvpartition(labels,'kfold',10); % Outputs: Trainsize and Testsize contain the size (= number of samples) of each train/test set

for j=1:4
    cp_labels = repartition(cp_labels);
    % Initialization of error vectors
    errClassLin = zeros(cp_labels.NumTestSets,1);

    % For a different testSet i each time
    for i = 1:cp_labels.NumTestSets
        % Attention,ici le cp_N.taining rend les INDICES des train samples
        % Quand trainIdx = 1 -> sample qui va dans le trainSet
        trainIdx = cp_labels.training(i);
        trainSet = features(trainIdx,:);
        trainLabels = labels(trainIdx);

        % Attention, ici le cp_N.test rend les INDICES des test samples
        % Quand testIdx = 1 -> sample va dans le testSet
        testIdx = cp_labels.test(i);
        testSet = features(testIdx,:);
        testLabels = labels(testIdx);

        % Calculus of class errors
        DiagQuadrclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype','diagquadratic');
        DiagQuadr_y = predict(DiagQuadrclassifier,testSet);
        errClassDiagQuadr(i) = classerrorOriginal(testLabels,DiagQuadr_y);
    end
    
    % Mean of errors
    cvErrDiagQuadr(j) = mean(errClassDiagQuadr);

    % Standard deviations of errors
    cvStdDiagQuadr(j) = std(errClassDiagQuadr);
end

% Plot
figure('Color','w');
title('10-fold cross Validation Error (changing repartition)');
hold on;
barwitherr(cvStdDiagQuadr, cvErrDiagQuadr);
ylabel('Class Error');
xticks([1 2 3 4]);
axis([0.5 4.5 0 0.6]);
xticklabels({'Partition 1','Partition 2','Partition 3','Partition 4'});
box off;
hold off;

C = cvpartition(labels,'LeaveOut'); %"leave-one-out" cross validation

for j=1:4
    C = repartition(C);

    % Error Calculation
    errClassLinC = zeros(C.NumTestSets,1);

    for i = 1:C.NumTestSets
        % Attention,ici le cp_N.taining rend les INDICES des train samples
        % Quand trainIdx = 1 -> sample qui va dans le trainSet
        trainIdxC = C.training(i);
        trainSetC = features(trainIdxC,:);
        trainLabelsC = labels(trainIdxC);

        % Attention, ici le cp_N.test rend les INDICES des test samples
        % Quand testIdx = 1 -> sample va dans le testSet
        testIdxC = C.test(i);
        testSetC = features(testIdxC,:);
        testLabelsC = labels(testIdxC);

        % Calculus of class errors
        DiagQuadrclassifierC = fitcdiscr(trainSetC,trainLabelsC,'discrimtype', 'diagquadratic');
        DiagQuadr_yC = predict(DiagQuadrclassifierC,testSetC);
        errClassDiagQuadrC(i) = classerrorOriginal(testLabelsC, DiagQuadr_yC);
    end

    % Mean of errors
    cvErrDiagQuadrC(j) = mean(errClassDiagQuadrC);
    
    % Standard deviations of errors
    cvStdDiagQuadrC(j) = std(errClassDiagQuadrC);
end

% Plot
figure('Color','w');
title('Leave-one out Cross Validation Error (changing repartition)');
hold on;
barwitherr(cvStdDiagQuadrC, cvErrDiagQuadrC);
ylabel('Class Error');
xticks([1 2 3 4]);
axis([0.5 4.5 -0.3 0.6]);
xticklabels({'Partition 1','Partition 2','Partition 3','Partition 4'});
box off;
hold off;
