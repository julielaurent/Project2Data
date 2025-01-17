close all;
clear all;
clc;


load('dataset_ERP.mat');

features = features(:,1:5:300);

%% Randomize the samples and Split Dataset
trainingPercentage = 50;

randFeatures = randperm(648);
trainSet = features(randFeatures(1:round(trainingPercentage/100*648)),:); % set 1
testSet = features(randFeatures(round(trainingPercentage/100*648)+1:end),:); % set 2
labelsTrain = labels(randFeatures(1:round(trainingPercentage/100*648)));
labelsTest = labels(randFeatures(round(trainingPercentage/100*648)+1:end));

%% DiagLinear classifier - training on training and testing sets

% Train diagonal linear classifier on training set/set1
DiagLinclassifierTrain = fitcdiscr(trainSet,labelsTrain,'discrimtype', 'diaglinear'); % LDA
% Train diagonal linear classifier on testing set/set2
DiagLinclassifierTest = fitcdiscr(testSet,labelsTest,'discrimtype', 'diaglinear'); % LDA

% Compute the error on TRAINING set/set1
% -> after classifier trained on training set % training error
DiagLin_yTrain1 = predict(DiagLinclassifierTrain,trainSet);
[errorClassificationDiagLinTrain1,errorClassDiagLinTrain1] = classerror(labelsTrain, DiagLin_yTrain1);
% -> after classifier trained on testing set % testing error
DiagLin_yTrain2 = predict(DiagLinclassifierTest,trainSet);
[errorClassificationDiagLinTrain2,errorClassDiagLinTrain2] = classerror(labelsTrain, DiagLin_yTrain2);

% Compute the error on TESTING set/set2
% -> after classifier trained on training set % testing error
DiagLin_yTest1 = predict(DiagLinclassifierTrain,testSet);
[errorClassificationDiagLinTest1,errorClassDiagLinTest1] = classerror(labelsTest, DiagLin_yTest1);
% -> after classifier trained on testing set % training error
DiagLin_yTest2 = predict(DiagLinclassifierTest,testSet);
[errorClassificationDiagLinTest2,errorClassDiagLinTest2] = classerror(labelsTest, DiagLin_yTest2);

%% Linear classifier - training on training and testing set

% Train linear classifier on training set/set1
LinclassifierTrain = fitcdiscr(trainSet,labelsTrain,'discrimtype', 'linear'); % LDA
% Train linear classifier on testing set/set2
LinclassifierTest = fitcdiscr(testSet,labelsTest,'discrimtype', 'linear'); % LDA

% Compute the error on TRAINING set/set1
% -> after classifier trained on training set % training error
Lin_yTrain1 = predict(LinclassifierTrain,trainSet);
[errorClassificationLinTrain1,errorClassLinTrain1] = classerror(labelsTrain, Lin_yTrain1);
% -> after classifier trained on testing set % testing error
Lin_yTrain2 = predict(LinclassifierTest,trainSet);
[errorClassificationLinTrain2,errorClassLinTrain2] = classerror(labelsTrain, Lin_yTrain2);

% Compute the error on TESTING set/set2
% -> after classifier trained on training set % testing error
Lin_yTest1 = predict(LinclassifierTrain,testSet);
[errorClassificationLinTest1,errorClassLinTest1] = classerror(labelsTest, Lin_yTest1);
% -> after classifier trained on testing set % training error
Lin_yTest2 = predict(LinclassifierTest,testSet);
[errorClassificationLinTest2,errorClassLinTest2] = classerror(labelsTest, Lin_yTest2);

%% DiagQuadr classifier - training on training and testing set
% Train diagonal quadratic classifier on training set/set1
DiagQuadrclassifierTrain = fitcdiscr(trainSet,labelsTrain,'discrimtype', 'diagquadratic'); % QDA
% Train linear classifier on testing set/set2
DiagQuadrclassifierTest = fitcdiscr(testSet,labelsTest,'discrimtype', 'diagquadratic'); % QDA

% Compute the error on TRAINING set/set1
% -> after classifier trained on training set % training error
DiagQuadr_yTrain1 = predict(DiagQuadrclassifierTrain,trainSet);
[errorClassificationDiagQuadrTrain1,errorClassDiagQuadrTrain1] = classerror(labelsTrain, DiagQuadr_yTrain1);
% -> after classifier trained on testing set % testing error
DiagQuadr_yTrain2 = predict(DiagQuadrclassifierTest,trainSet);
[errorClassificationDiagQuadrTrain2,errorClassDiagQuadrTrain2] = classerror(labelsTrain, DiagQuadr_yTrain2);

% Compute the error on TESTING set/set2
% -> after classifier trained on training set % testing error
DiagQuadr_yTest1 = predict(DiagQuadrclassifierTrain,testSet);
[errorClassificationDiagQuadrTest1,errorClassDiagQuadrTest1] = classerror(labelsTest, DiagQuadr_yTest1);
% -> after classifier trained on testing set % training error
DiagQuadr_yTest2 = predict(DiagQuadrclassifierTest,testSet);
[errorClassificationDiagQuadrTest2,errorClassDiagQuadrTest2] = classerror(labelsTest, DiagQuadr_yTest2);

%% Quadratic classifier - training on training and testing set

% % Train linear classifier on training set
% QuadrclassifierTrain = fitcdiscr(trainSet,labelsTrain,'discrimtype', 'quadratic'); % LDA
% % Train linear classifier on testing set
% QuadrclassifierTest = fitcdiscr(testSet,labelsTest,'discrimtype', 'quadratic'); % LDA
% 
% % Compute the error on TRAINING set
% % -> after classifier trained on training set % training error
% Quadr_yTrain1 = predict(QuadrclassifierTrain,trainSet);
% [errorClassificationQuadrTrain1,errorClassQuadrTrain1] = classerror(labelsTrain, Quadr_yTrain1);
% % -> after classifier trained on testing set % testing error
% Quadr_yTrain2 = predict(QuadrclassifierTest,trainSet);
% [errorClassificationQuadrTrain2,errorClassQuadrTrain2] = classerror(labelsTrain, Quadr_yTrain2);
% 
% % Compute the error on TESTING set
% % -> after classifier trained on training set % training error
% Quadr_yTest1 = predict(QuadrclassifierTrain,testSet);
% [errorClassificationQuadrTest1,errorClassQuadrTest1] = classerror(labelsTest, Quadr_yTest1);
% % -> after classifier trained on testing set % testing error
% Quadr_yTest2 = predict(QuadrclassifierTest,testSet);
% [errorClassificationQuadrTest2,errorClassQuadrTest2] = classerror(labelsTrain, Quadr_yTest2);


% Error, singular matrix


%% Graphs

% Testing on set 1 (training set)
% Training with set 1 (training set) and 2 (testing set)
figure('Color','w');
title('Error on Set 1 Classification');
hold on;
bar([errorClassDiagLinTrain1,errorClassDiagLinTrain2; errorClassLinTrain1,errorClassLinTrain2; errorClassDiagQuadrTrain1,errorClassDiagQuadrTrain2]);
legend('Classifier trained on Set 1 (Training Error)','Classifier Trained on Set 2 (Testing Error)');
ylabel('Error');
xticks([1 2 3]);
xticklabels({'DiagLinear','Linear','DiagQuadratic'});
axis([0.5 3.5 0 0.4]);
box off;
hold off;

% Testing on set 2 (testing set)
% Training with set 1 (training set) and 2 (testing set)
figure('Color','w');
title('Error on Set 2 Classification');
hold on;
bar([errorClassDiagLinTest2,errorClassDiagLinTest1; errorClassLinTest2,errorClassLinTest1; errorClassDiagQuadrTest2,errorClassDiagQuadrTest1]);
legend('Classifier trained on Set 2 (Training Error)','Classifier Trained on Set 1 (Testing Error)');
ylabel('Error');
xticks([1 2 3]);
xticklabels({'DiagLinear','Linear','DiagQuadratic'});
axis([0.5 3.5 0 0.4]);
box off;
hold off;


