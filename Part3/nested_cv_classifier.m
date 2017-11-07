close all;
clear all;
clc;

load('dataset_ERP.mat');


%% k-fold cross-validation with Fischer

Nfeature = 60; % number of features tried
nModel = 0; 
N = Nfeature * 3; %total number of models
Kout = 3; %number of outer loop folds
Kin = 10; %number of inner folds
classifierType = {'diaglinear','linear','diagquadratic'};

model = struct('classifier',[],'number_of_features',[]);

% Rank of features
[orderedInd, orderedPower] = rankfeat(features,labels,'fisher');

% Outer partition
cp_labels_out = cvpartition (labels,'kfold',Kout);

% Initialization
 errTest_out = zeros(1,Kout); 
 errTrain_out = zeros(1,Kout); 
 validationerr_in = zeros(N,Kin); 
 errTrain_in = zeros(N,Kin); 
 bestModel_in = zeros(1,Kout); 
 optimal_validationerror_in  = zeros(1,Kout); 
 optimal_trainingerror_in  = zeros(1,Kout); 

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
    
    %number of models tried (60 features * 3 classifiers = 180)
    %model j: classifier 1 with 1 feature, classifier 1 with 2 features,
    %..., classifier 1 with 60 features, classifier 2 with 1 feature, ...,
    %classifier 3 with 60 features
    % The 60 first models are from the diagonal linear classifier
    % Models 61:120 are form the linear classifier
    % The last 60 models are from the diagonal quadratic classifier
    %for j = 1:N
       
       %Choice of classifier
       for type = 1:3
           c = char(classifierType(type));
           %Choice of the number of features
           for nbF = 1:Nfeature
               nModel = nModel + 1;
               features_model_in = [features_model_in, features(trainIdx_out,orderedInd(nbF))];
               model(nModel).classifier = c;
               model(nModel).number_of_features = nbF;
               
                   % Kin-fold cv for each model (one model is one
                   % classifier associated to one number of feature
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
                       classifier_in = fitcdiscr(trainSet_in,trainLabels_in,'discrimtype', c);

                       % Calculus of class error on test set -> validation testing error (NxKin)
                       yTest_in = predict(classifier_in,testSet_in);
                       validationerr_in(nModel,i) = classerror(testLabels_in, yTest_in);
          
                       % Calculus of class error on train set -> training error (NxKin)
                       yTrain_in = predict(classifier_in,trainSet_in);
                       errTrain_in(nModel,i) = classerror(trainLabels_in, yTrain_in);
                       
                   end    
           end
    
       end
    %end
   
    % Best validation error and total model
    mean_validationerror_in = mean(validationerr_in,2);
    optimal_validationerror_in(p) = min(mean_validationerror_in);
    mean_trainingerror_in = mean(errTrain_in,2);
    optimal_trainingerror_in(p) = min(mean_trainingerror_in);
    bestModelNumber = find(mean_validationerror_in == optimal_validationerror_in(p));
    bestModel_in(p) = bestModelNumber(1); % Si plusieurs min egaux, je choisis le premier
    
%     %Calculus of means of errors per classifier and best number of features per
%     %classifier
%     mean_validationerror_diaglin = mean(validationerr_in(1:Nfeature,:),2);
%     optimal_validationerror_diaglin(p) = min(mean_validationerror_diaglin);
%     mean_trainingerror_diaglin = mean(errTrain_in(1:Nfeature,:),2);
%     optimal_trainingerror_diaglin(p) = min(mean_trainingerror_diaglin);
%     bestFeatureNumberDiaglin = model(find(mean_validationerror_diaglin == optimal_validationerror_diaglin(p))).number_of_features;
%     bestFeatureNumberDiaglin_in(p) = bestFeatureNumberDiaglin(1);
%     
%     mean_validationerror_lin = mean(validationerr_in(Nfeature+1:2*Nfeature,:),2);
%     optimal_validationerror_lin(p) = min(mean_validationerror_lin);
%     mean_trainingerror_lin = mean(errTrain_in(Nfeature+1:2*Nfeature,:),2);
%     optimal_trainingerror_lin(p) = min(mean_trainingerror_lin);
%     bestFeatureNumberLin = model(find(mean_validationerror_lin == optimal_validationerror_lin(p))).number_of_features;
%     bestFeatureNumberLin_in(p) = bestFeatureNumberLin(1);
%     
%     mean_validationerror_diagquadratic = mean(validationerr_in(2*Nfeature+1:end),2);
%     optimal_validationerror_diagquadratic(p) = min(mean_validationerror_diagquadratic);
%     mean_trainingerror_diagquadratic = mean(errTrain_in(2*Nfeature+1:end),2);
%     optimal_trainingerror_diagquadratic(p) = min(mean_trainingerror_diagquadratic);
%     bestFeatureNumberDiagquadratic = model(find(mean_validationerror_diagquadratic == optimal_validationerror_diagquadratic(p))).number_of_features;
%     bestFeatureNumberDiagquadratic_in(p) = bestFeatureNumberDiagquadratic(1);
    
    % Construct our data matrix with the selected number of features
    % Extract best model data 
    bestModelClassifier = model(bestModelNumber).classifier;
    for j = 1:model(bestModel_in(p)).number_of_features
       features_model_out = [features_model_out, features(:,orderedInd(j))];
    end
     
    % Select the train and test data for the outer fold
    trainSet_out = features_model_out(trainIdx_out,:); 
    testSet_out = features_model_out(testIdx_out,:);
       
    % Classifier construction
    classifier_out = fitcdiscr(trainSet_out,trainLabels_out,'discrimtype', bestModelClassifier);

    % Calculus of class error on test set -> testing error (1xKout)
    yTest_out = predict(classifier_out,testSet_out);
    errTest_out(p) = classerror(testLabels_out, yTest_out);
    
    %%%%% Calculer la test error pour chaque classifiermais je sais pas
    %%%%% comment faire
        
    
    % Calculus of class error on train set -> training error (1xKout)
    %yTrain_out = predict(classifier_out,trainSet_out);
    %errTrain_out(p) = classerror(trainLabels_out, yTrain_out);
    
end

%Calculus of best model characteristics
model(bestModel_in)

%boxplots of distributions
figure('Color','w');
l = zeros(9,1);
l(1:Kout) = 1;
l(Kout+1:2*Kout) = 2;
l(2*Kout+1:3*Kout) = 3;
subplot(3,1,1);
boxplot([optimal_trainingerror_diaglin; optimal_validationerror_diaglin; errTest_out], l, 'Labels',{'Optimal Training','Optimal Validation','Test'});
box off;
ylabel('Error');
title('Error Distributions for Diagonal Linear Classifier')
subplot(3,1,2);
boxplot([optimal_trainingerror_lin; optimal_validationerror_lin; errTest_out], l, 'Labels',{'Optimal Training','Optimal Validation','Test'});
box off;
ylabel('Error');
title('Error Distributions for Linear Classifier')
subplot(3,1,3);
boxplot([optimal_trainingerror_diagquadratic; optimal_validationerror_diagquadratic; errTest_out], l, 'Labels',{'Optimal Training','Optimal Validation','Test'});
box off;
ylabel('Error');
title('Error Distributionsfor DIagonal Quadratic Classifier')