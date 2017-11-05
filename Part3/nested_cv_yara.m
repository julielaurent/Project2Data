close all;
clear all;
clc;

load('dataset_ERP.mat');


%% k-fold cross-validation with Fischer

N = 60; %number of models tried
Kout = 3; %number of outer loop folds
Kin = 4; %number of inner folds

% Rank of features
[orderedInd, orderedPower] = rankfeat(features,labels,'fisher');

% Outer partition
cp_labels_out = cvpartition (labels,'kfold',Kout);

% Initialization
 errTest_out = zeros(1,Kout); 
 errTrain_out = zeros(1,Kout); 
 validationerr_in = zeros(N,Kin); 
 errTrain_in = zeros(N,Kin); 
 nbfeature_minTesterror_in = zeros(1,Kout); 
 min_validationerror_in  = zeros(1,Kout);
 min_errTrain_in = zeros(1,Kout);

for p = 1:Kout
    features_model_out = [];
    
    % Attention,ici le cp_N.training rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx_out = cp_labels_out.training(p);
    testIdx_out = cp_labels_out.test(p);
    trainLabels_out = labels(trainIdx_out);
    testLabels_out = labels(testIdx_out);
    
    % Inner partition on the train set of our outer-fold
    cp_labels_in = cvpartition (trainLabels_out,'kfold',Kin);
    
    features_model_in = [];
    
    for j = 1:N
       features_model_in = [features_model_in, features(trainIdx_out,orderedInd(j))];
       
       % Kin-fold cv for each model with a different number of features
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
          
          % Calculus of class error on train set -> training error (NxKin)
          yTrain_in = predict(DiagLinclassifier_in,trainSet_in);
          errTrain_in(j,i) = classerror(trainLabels_in, yTrain_in);
       end    
    end
    
    % Best number of features according to inner cross-validation
    mean_validationerror_in = mean(validationerr_in,2);
    min_validationerror_in(p) = min(mean_validationerror_in);
    nbfeature_minTesterror = find(mean_validationerror_in == min_validationerror_in(p));
    nbfeature_minTesterror_in(p) = nbfeature_minTesterror(1); % Si plusieurs min egaux, je choisis le premier
    mean_errTrain_in = mean(errTrain_in,2);
    min_errTrain_in(p) = min(mean_errTrain_in);
    
    % Construct our data matrix with the selected number of features
    for j = 1:nbfeature_minTesterror_in(p)
       features_model_out = [features_model_out, features(:,orderedInd(j))];
    end
     
    % Select the train and test data for the outer fold
    trainSet_out = features_model_out(trainIdx_out,:); 
    testSet_out = features_model_out(testIdx_out,:);
       
    % Classifier construction
    DiagLinclassifier_out = fitcdiscr(trainSet_out,trainLabels_out,'discrimtype', 'diagLinear');

    % Calculus of class error on test set -> testing error (1xKout)
    yTest_out = predict(DiagLinclassifier_out,testSet_out);
    errTest_out(p) = classerror(testLabels_out, yTest_out);
    
    % Calculus of class error on train set -> training error (1xKout)
    yTrain_out = predict(DiagLinclassifier_out,trainSet_out);
    errTrain_out(j,i) = classerror(trainLabels_out, yTrain_out);
    
end

%boxplots of distributions
figure('Color','w');
subplot(2,2,1)
boxplot(min_errTrain_in);
box off;
title('Boxplot of optimal training error')
subplot(2,2,2)
boxplot(min_validationerror_in);
box off;
title('Boxplot of optimal validation error')
subplot(2,2,3)
boxplot(errTest_out);
box off;
title('Boxplot of test error')

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
% xticks(0:5:60);
% box off;
% hold off;