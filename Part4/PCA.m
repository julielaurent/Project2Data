close all;
clear all;
clc;

load('dataset_ERP.mat');

%% PCA
% explained = percentage of total variance explained by each PC
% mu = estimated mean of each variable (features) in dataset "features"
[coeff, score, variance, tsquared, explained, mu] = pca(features, 'Centered', false); %A v?rifier centered data ou pas

%% Covariance matrix
covMatrix_original = cov(features);
covMatrix_pca = cov(features * coeff); %ou alors covMatrix_pca = cov(score);

% Matrices with only covariances (off-diagonal elements only, diagonal elements = 0)
covMatrix_original_cov =  covMatrix_original;
covMatrix_pca_cov = covMatrix_pca;

% Calculs covariances original
for i = 1:2400
    for j = 1:2400
        if i == j
            covMatrix_original_cov(i,j) = 0;
        end
    end
end

% Calculs covariances pca
for i = 1:648
    for j = 1:648
        if i == j
            covMatrix_pca_cov(i,j) = 0;
        end
    end
end

% Maximal covariance value for original data
maxCovValue_original = max(max(covMatrix_original_cov));
maxCovValue_pca = max(max(covMatrix_pca_cov));

%% Figures
figure('Color','w');
imshow(covMatrix_original); %2400*2400
c = colorbar;
c.Label.String = 'Normalized Matrix Values';
title('Original Covariance Matrix (2400*2400)');

figure('Color','w');
imshow(covMatrix_pca); %648*648
c = colorbar;
c.Label.String = 'Normalized Matrix Values';
title('Covariance Matrix after PCA (648*648)');

%% Cumulative variance
% First element representes the percentage of variance explained by PC1
% Second element represents the percentage of variance explained by the 2
% first PCs (PC1 & PC2)
% And so on...
VarCumPercentage = cumsum(variance)/sum(variance)*100;
figure('Color','w');
title('Number of PCA Explaining 90% of the Total Variance of the Data');
plot(VarCumPercentage);
xlabel('Principal Components');
ylabel('Cumulative Variance Explained [%]');
box off;
axis([0 648 0 100]);
line([0 648], [90 90],'Color','r');
line([44 44],[0 90],'Color','r','LineStyle','--');

% 44 features explain 90.1966 % of the variances

%% Cross-validation with PCA (a voir semaine pro)

% Partition: K = 10
cp_labels = cvpartition (labels,'kfold',10);

% PCA 
[coeff, score, variance] = pca(features, 'Centered', false); %A v?rifier centered data ou pas 
%[coeff, score, variance] = pca(features(trainIdx,:), 'Centered', false);
%score_uncentered = features*coeff;

% Initialization of error vector
N = 60;
errClassDiagLinTest = zeros(N, cp_labels.NumTestSets);
errClassDiagLinTrain = zeros(N,cp_labels.NumTestSets);

% Cross-validation
for i = 1:cp_labels.NumTestSets
    
    % Initialisation
    features_model = [];
      
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx = cp_labels.training(i);
    trainLabels = labels(trainIdx);
    testIdx = cp_labels.test(i);
    testLabels = labels(testIdx);
    
   for j = 1:N
       
        %features_model = [features_model, score(:,j)]; % je mettrai pas score parce que c'est centr?, donc plutot: features(:,j) * coeff(j,:)
        features_model = [features_model, score(:,j)];
        
        trainSet = features_model(trainIdx,:);
        testSet = features_model(testIdx,:);
         
        % Classifier construction
        DiagLinclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagLinear');

        % Calculus of class error on test set -> testing error (NxK)
        DiagLin_yTest = predict(DiagLinclassifier,testSet);
        errClassDiagLinTest(j,i) = classerror(testLabels, DiagLin_yTest);
        
        % Calculus of class error on train set -> training error (NxK)
        DiagLin_yTrain = predict(DiagLinclassifier,trainSet);
        errClassDiagLinTrain(j,i) = classerror(trainLabels, DiagLin_yTrain);
    end
end    

% Best number of PCs with min test error
meanTesterror = mean(errClassDiagLinTest,2);
minerror = min(meanTesterror);
nbPC_minTesterror = find(meanTesterror == minerror);
nbPC_minTesterror = nbPC_minTesterror(1) % If several min value, select the first one

figure('Color','w');
plot(errClassDiagLinTest,'g--');
hold on;
plot(meanTesterror,'g','LineWidth',2);
hold on;
line([44 44],[0 1],'Color','k','LineStyle','-'); %number of PCs explaining 90% of the original features
axis([0 60 0 0.5]);
xlabel('Number of PCs'); ylabel('Error');
title('Error per fold with a PCA');
box off;
hold off;

%% Normalization of data (a voir semaine pro) -> Normaliser avant tout?
% features_normalized = zscore (features);
% [coeff_n, score_n, variance_n] = pca(features_normalized, 'Centered', false); %A v?rifier centered data ou pas
% 
% pc_normalized = zscore(score); %score ou features*coeff ?