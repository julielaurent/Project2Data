close all;
clear all;
clc;

load('dataset_ERP.mat');

size_labels = size(labels);
% Total number of samples
N = size_labels(1);
% CV partition
cp_labels = cvpartition (labels,'kfold',10);
% Options
opt = statset('Display','iter','MaxIter',100);
% Classifier
classifierType = 'diaglinear';

% defines the criterion used to select features and to determine when to stop
fun = @(xT,yT,xt,yt) length(yt)*(your_error(yt,predict(fitcdiscr(xT,yT,'discrimtype',classifierType),xt)));

% Initialization of error vectors
errClassDiagLin = zeros(cp_labels.NumTestSets,1);

% For a different testSet i each time -> Outer loop
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
    
    % sel is logical vector indicating which features are finally chosen
    % hst is a scalar strucure with the fiels Crit (vector containing criterion
    % values at each step) and In (logical matrix -> row indicates feature
    % selected at each step)
    % PERFORMS A 10-FOLD CV FOR EACH CANDIDATE FEATURE SUBSET
    [sel,hst] = sequentialfs(fun,trainSet,trainLabels,'cv',cp,'options',opt);
    
    opt_validationError = hst.Crit(end);
    
    % Calculus of class errors
    DiagLinclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagLinear');
    DiagLin_y = predict(DiagLinclassifier,testSet);
    errClassDiagLin(i) = classerrorOriginal(testLabels, DiagLin_y);

    
end

% % Mean of errors
cvErrDiagLin = mean(errClassDiagLin);

% Standard deviations of errors
cvStdDiagLin = std(errClassDiagLin);
