close all;
clear all;
clc;

load('dataset_ERP.mat');

size_labels = size(labels);
% Total number of samples
N = size_labels(1);
% CV partition
cp_labels_out = cvpartition (labels,'kfold',3);
% Options
opt = statset('Display','iter','MaxIter',100);
% Classifier
classifierType = 'diaglinear';
% defines the criterion used to select features and to determine when to stop
% si j'ai bien compris,x -> set, y -> labels, t -> test/validation, T -> train
fun = @(xT,yT,xt,yt) length(yt)*(classerror(yt,predict(fitcdiscr(xT,yT,'discrimtype',classifierType),xt)));

% Initialization of error vectors
errClassDiagLin = zeros(cp_labels_out.NumTestSets,1);
opt_validationError = zeros(cp_labels_out.NumTestSets,1);
nb_selectedFeatures = zeros(cp_labels_out.NumTestSets,1);

% For a different testSet i each time
% OUTER CV
for i = 1:cp_labels_out.NumTestSets
    
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx = cp_labels_out.training(i);
    trainSet = features(trainIdx,:);
    trainLabels = labels(trainIdx);
   
    % Attention, ici le cp_N.test rend les INDICES des test samples
    % Quand testIdx = 1 -> sample va dans le testSet
    testIdx = cp_labels_out.test(i);
    testSet = features(testIdx,:);
    testLabels = labels(testIdx);
    
    % Inner partition
    cp_labels_in = cvpartition(trainLabels,'kfold',10);
    
    % sel is logical vector indicating which features are finally chosen
    % hst is a scalar strucure with the fiels Crit (vector containing criterion
    % values at each step) and In (logical matrix -> row indicates feature
    % selected at each step)
    % PERFORMS A 10-FOLD CV FOR EACH CANDIDATE FEATURE SUBSET
    [sel,hst] = sequentialfs(fun,trainSet,trainLabels,'cv',cp_labels_in,'options',opt,'keepout',[1:300,2000:2400]);
    
    opt_validationError(i) = hst.Crit(end);
    nb_selectedFeatures(i) = find(hst.Crit == opt_validationError(i));
    trainSet_selectedFeatures = trainSet(:,sel);
    testSet_selectedFeatures = testSet(:,sel);
    selectedFeatures = find(sel);
    
    % Calculus of class errors
    DiagLinclassifier = fitcdiscr(trainSet_selectedFeatures,trainLabels,'discrimtype', 'diagLinear');
    DiagLin_y = predict(DiagLinclassifier,testSet_selectedFeatures);
    errClassDiagLin(i) = classerrorOriginal(testLabels, DiagLin_y);

end

%ONLY TEST ERROR
% Mean of errors
cvErrDiagLin = mean(errClassDiagLin);
% Standard deviations of errors
cvStdDiagLin = std(errClassDiagLin);
% Minimal error
cvErrMin = min(cvErrDiagLin);
% Number of features with minimal error
nb_features = find(cvErrDiagLin == cvErrMin);