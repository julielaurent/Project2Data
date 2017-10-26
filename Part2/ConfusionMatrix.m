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


%% Confusion matrix
%  CM = confusionmat(G,GHAT) returns the confusion matrix CM determined by 
%     the known group labels G and the predicted group labels GHAT. G and
%     GHAT are grouping variables with the same number of observations.


DiagQuadrclassifier = fitcdiscr(trainSet,labelsTrain ,'discrimtype', 'diagQuadratic');
DiagQuadr_y = predict(DiagQuadrclassifier,testSet);

CM = confusionmat(labelsTest,DiagQuadr_y);
