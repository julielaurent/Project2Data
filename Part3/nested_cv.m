close all;
clear all;
clc;

load('dataset_ERP.mat');


%% k-fold cross-validation with Fischer

N = 60; %number of models tried
Kout = 10; %number of outer loop folds
Kin = 10; %number of inner folds

% Outer partition
cp_labels_out = cvpartition (labels,'kfold',Kout);

% Initialization
 errTest_out = zeros(1,Kout); 
 errTrain_out = zeros(1,Kout); 
 validationerr_in = zeros(N,Kin); 
 errTrain_in = zeros(N,Kin); 
 nbfeature_minTesterror_in = zeros(1,Kout); 
 min_validationerror_in  = zeros(1,Kout); 
 min_trainingerror_in  = zeros(1,Kout); 

for p = 1:Kout
    trainSet_out = [];
    testSet_out = [];
    
    % Attention,ici le cp_N.training rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx_out = cp_labels_out.training(p);
    testIdx_out = cp_labels_out.test(p);
    trainLabels_out = labels(trainIdx_out);
    testLabels_out = labels(testIdx_out);
    train_out = features(trainIdx_out,:);
    test_out = features(testIdx_out,:);
    
    % Rank of features for outer loop, on training set
    [orderedIndout, orderedPowerout] = rankfeat(train_out,trainLabels_out,'fisher');
    
    % Inner partition on the train set of our outer-fold
    cp_labels_in = cvpartition (trainLabels_out,'kfold',Kin);
    
    % Kin-fold 
    for i = 1:Kin
         trainSet_in = [];
         testSet_in = [];
         
         % Attention,ici le cp_N.taining rend les INDICES des train samples
         % Quand trainIdx = 1 -> sample qui va dans le trainSet
         trainIdx_in = cp_labels_in.training(i);
         trainLabels_in = trainLabels_out(trainIdx_in);
         testIdx_in = cp_labels_in.test(i);
         testLabels_in = trainLabels_out(testIdx_in);
         train_in = train_out(trainIdx_in,:);
         test_in = train_out(testIdx_in,:);
        
         % Rank of features for inner loop, on training set
         [orderedIndin, orderedPowerin] = rankfeat(train_in,trainLabels_in,'fisher');
         
         % Test different models with this inner fold cv
          for j = 1:N
            trainSet_in = [trainSet_in, train_in(:,orderedIndin(j))]; 
            testSet_in = [testSet_in, test_in(:,orderedIndin(j))];

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
    mean_trainingerror_in = mean(errTrain_in,2);
    min_trainingerror_in(p) = min(mean_trainingerror_in);
    nbfeature_minTesterror = find(mean_validationerror_in == min_validationerror_in(p));
    nbfeature_minTesterror_in(p) = nbfeature_minTesterror(1); % Si plusieurs min egaux, je choisis le premier
    
    % Construct our data matrix with the selected number of features on the
    % ranking done one the training set of the outer fold
    for j = 1:nbfeature_minTesterror_in(p)
       trainSet_out = [trainSet_out, train_out(:,orderedIndout(j))];
       testSet_out = [testSet_out, test_out(:,orderedIndout(j))];
    end
     
    % Classifier construction
    DiagLinclassifier_out = fitcdiscr(trainSet_out,trainLabels_out,'discrimtype', 'diagLinear');

    % Calculus of class error on test set -> testing error (1xKout)
    yTest_out = predict(DiagLinclassifier_out,testSet_out);
    errTest_out(p) = classerror(testLabels_out, yTest_out);
    
    % Calculus of class error on train set -> training error (1xKout)
    yTrain_out = predict(DiagLinclassifier_out,trainSet_out);
    errTrain_out(p) = classerror(trainLabels_out, yTrain_out);
    
end

%Calculus of median to compare with previous code
median_nbfeatures = median(nbfeature_minTesterror_in);

%boxplots of distributions
figure('Color','w');
l = zeros(15,1);
l(1:Kout) = 1;
l(Kout+1:2*Kout) = 2;
l(2*Kout+1:3*Kout) = 3;
boxplot([min_trainingerror_in, min_validationerror_in, errTest_out], l, 'Labels',{'Optimal Training','Optimal Validation','Test'});
box off;
ylabel('Error');
title('Boxplots of Error Distributions')