close all;
clear all;
clc;

load('dataset_ERP.mat');


%% k-fold cross-validation with Fischer

N = 60;
Kout = 3;
Kin = 4;

% Rank of features
[orderedInd, orderedPower] = rankfeat(features,labels,'fisher');

% Outer partition
cp_labels_out = cvpartition (labels,'kfold',Kout);

% Initialization
 %errClassDiagLinTest = zeros(N, cp_labels.NumTestSets);
 %errClassDiagLinTrain = zeros(N,cp_labels.NumTestSets);
 errTrain_out = []; % ? changer apr?s
 errTest_out = []; % ? changer apr?s
 errTrain_in = []; % ? changer apr?s
 validationerrTest_in = []; % ? changer apr?s
 

for p = 1:Kout
    features_model_out = [];
    
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx_out = cp_labels_out.training(p);
    trainLabels_out = labels(trainIdx_out);

    % Attention, ici le cp_N.test rend les INDICES des test samples
    % Quand testIdx = 1 -> sample va dans le testSet
    testIdx_out = cp_labels_out.test(p);
    testLabels_out = labels(testIdx_out);
    
    % Inner partition
    cp_labels_in = cvpartition (trainLabels_out,'kfold',Kin);
    
    features_model_in = [];
    
    for j = 1:N
       features_model_in = [features_model_in, features(trainIdx_out,orderedInd(j))];
       % For a different testSet i each time
       for i = 1:Kin

          % Attention,ici le cp_N.taining rend les INDICES des train samples
          % Quand trainIdx = 1 -> sample qui va dans le trainSet
          trainIdx_in = cp_labels_in.training(i);
          trainSet_in = features_model_in(trainIdx_in,:);
          trainLabels_in = trainLabels_out(trainIdx_in);

          % Attention, ici le cp_N.test rend les INDICES des test samples
          % Quand testIdx = 1 -> sample va dans le testSet
          testIdx_in = cp_labels_in.test(i);
          testSet_in = features_model_in(testIdx_in,:);
          testLabels_in = trainLabels_out(testIdx_in);

          % Classifier construction
          DiagLinclassifier_in = fitcdiscr(trainSet_in,trainLabels_in,'discrimtype', 'diagLinear');

          % Calculus of class error on test set -> validation testing error (NxKin)
          yTest_in = predict(DiagLinclassifier_in,testSet_in);
          validationerr_in(j,i) = classerror(testLabels_in, yTest_in);
       end    
    end
    
    % Best number of features according to inner cross-validation
    mean_validationerror_in = mean(validationerr_in,2);
    min_validationerror_in = min(mean_validationerror_in);
    nbfeature_minTesterror_in = find(mean_validationerror_in == min_validationerror_in);
    
    % Construct our data matrix with the selected number of features
    for j = 1:nbfeature_minTesterror_in
       features_model_out = [features_model_out, features(:,orderedInd(j))];
    end
       
    trainSet_out = features_model_out(trainIdx_out,:); 
    testSet_out = features_model_out(testIdx_out,:);
       
    % Classifier construction
    DiagLinclassifier_out = fitcdiscr(trainSet_out,trainLabels_out,'discrimtype', 'diagLinear');

    % Calculus of class error on test set -> testing error (NxK)
    yTest_out = predict(DiagLinclassifier_out,testSet_out);
    errTest_out(j,i) = classerror(testLabels_out, yTest_out);

    % Calculus of class error on train set -> training error (NxK)
    yTrain_out = predict(DiagLinclassifier_out,trainSet_out);
    errTrain_out(j,i) = classerror(trainLabels_out, yTrain_out);
end