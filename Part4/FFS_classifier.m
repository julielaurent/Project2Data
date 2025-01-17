close all;
clear all;
clc;

load('dataset_ERP.mat');

%% Nested cross-validation with forward feature selection and classifier selection

Kout = 10; %number of outer loop folds
Kin = 10; %number of inner folds

% Options
opt = statset('Display','iter','MaxIter',100);

classifierType = {'diaglinear','linear','diagquadratic'};

% Outer partition
cp_labels_out = cvpartition (labels,'kfold',Kout);

% Initialization
 errTest_out = zeros(1,Kout); 
 errTrain_out = zeros(1,Kout); 
 bestClassifierType = zeros(1,Kout); 
 errClassTest_in = zeros(cp_labels_out.NumTestSets,1);
 opt_validationErrorDL = zeros(cp_labels_out.NumTestSets,1);
 opt_validationErrorL = zeros(cp_labels_out.NumTestSets,1);
 opt_validationErrorDQ = zeros(cp_labels_out.NumTestSets,1);
 nb_selectedFeaturesDL = zeros(cp_labels_out.NumTestSets,1);
 nb_selectedFeaturesL = zeros(cp_labels_out.NumTestSets,1);
 nb_selectedFeaturesDQ = zeros(cp_labels_out.NumTestSets,1);
 
 minTestErr = zeros(cp_labels_out.NumTestSets,1);
 bestModelClassifier = zeros(cp_labels_out.NumTestSets,1);
 bestNbFeatures = zeros(cp_labels_out.NumTestSets,1);
 bestSelectedFeatures = zeros(cp_labels_out.NumTestSets,1);
 
for p = 1:Kout
    
    % Attention,ici le cp_N.training rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx = cp_labels_out.training(p);
    testIdx = cp_labels_out.test(p);
    trainSet = features(trainIdx,:);
    testSet = features(testIdx,:);
    trainLabels = labels(trainIdx);
    testLabels = labels(testIdx);

    cp_labels_in = cvpartition (trainLabels,'kfold',Kin);
         
    funDL = @(xT,yT,xt,yt) length(yt)*(classerror(yt,predict(fitcdiscr(xT,yT,'discrimtype','diaglinear'),xt)));
    [selDL,hstDL] = sequentialfs(funDL,trainSet,trainLabels,'cv',cp_labels_in,'options',opt,'keepout',[1:300,2000:2400]);
    opt_validationErrorDL(p) = hstDL.Crit(end);
    nb_selectedFeaturesDL(p) = find(hstDL.Crit == opt_validationErrorDL(p));
              
    funL = @(xT,yT,xt,yt) length(yt)*(classerror(yt,predict(fitcdiscr(xT,yT,'discrimtype','linear'),xt)));
    [selL,hstL] = sequentialfs(funL,trainSet,trainLabels,'cv',cp_labels_in,'options',opt,'keepout',[1:300,2000:2400]);
    opt_validationErrorL(p) = hstL.Crit(end);       
    nb_selectedFeaturesL(p) = find(hstL.Crit == opt_validationErrorL(p));
    
    funDQ = @(xT,yT,xt,yt) length(yt)*(classerror(yt,predict(fitcdiscr(xT,yT,'discrimtype','diagquadratic'),xt)));
    [selDQ,hstDQ] = sequentialfs(funDQ,trainSet,trainLabels,'cv',cp_labels_in,'options',opt,'keepout',[1:300,2000:2400]);    
    opt_validationErrorDQ(p) = hstDQ.Crit(end);
    nb_selectedFeaturesDQ(p) = find(hstDQ.Crit == opt_validationErrorDQ(p));
    
    min_optimalValErr = min([opt_validationErrorDL(p) opt_validationErrorL(p) opt_validationErrorDQ(p)]);
    
    if min_optimalValErr == opt_validationErrorDL(p)
        sel = selDL;
        bestNbFeatures(p) = nb_selectedFeaturesDL(p);
        bestClassifierType(p) = 1;
    elseif min_optimalValErr == opt_validationErrorL(p)
        sel = selL;
        bestNbFeatures(p) = nb_selectedFeaturesL(p);
        bestClassifierType(p) = 2;
    elseif min_optimalValErr == opt_validationErrorDQ(p)
        sel = selDQ;
        bestNbFeatures(p) = nb_selectedFeaturesDQ(p);
        bestClassifierType(p) = 3;
    end
    
    trainSet_selectedFeatures = trainSet(:,sel);
    testSet_selectedFeatures = testSet(:,sel);
    
    classifier = fitcdiscr(trainSet_selectedFeatures,trainLabels,'discrimtype', char(classifierType(bestClassifierType(p))));
    
    yTest_out = predict(classifier,testSet_selectedFeatures);
    errTest_out(p) = classerror(testLabels, yTest_out);
    
    yTrain_out = predict(classifier,trainSet_selectedFeatures);
    errTrain_out(p) = classerror(trainLabels, yTrain_out);
    
end