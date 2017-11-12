close all;
clear all;
clc;

load('dataset_ERP.mat');

% %% PCA
% % explained = percentage of total variance explained by each PC
% % mu = estimated mean of each variable (features) in dataset "features"
% [coeff, score, variance, tsquared, explained, mu] = pca(features, 'Centered', false); %A v?rifier centered data ou pas
% 
% %% Covariance matrix
% covMatrix_original = cov(features);
% covMatrix_pca = cov(features * coeff); %ou alors covMatrix_pca = cov(score);
% 
% % Matrices with only covariances (off-diagonal elements only, diagonal elements = 0)
% covMatrix_original_cov = cov(features);
% covMatrix_pca_cov = cov(features * coeff);
% 
% % Calculs covariances original
% for i = 1:2400
%     for j = 1:2400
%         if i == j
%             covMatrix_original_cov(i,j) = 0;
%         end
%     end
% end
% 
% % Calculs covariances pca
% for i = 1:648
%     for j = 1:648
%         if i == j
%             covMatrix_pca_cov(i,j) = 0;
%         end
%     end
% end
% 
% % Maximal covariance value for original data
% maxCovValue_original = max(max(covMatrix_original_cov));
% maxCovValue_pca = max(max(covMatrix_pca_cov));
% 
% %% Figures
% %figure('Color','w');
% %imshow(covMatrix_original);
% 
% %figure('Color','w');
% %imshow(covMatrix_pca);
% 
% %% Cumulative variance
% % First element representes the percentage of variance explained by PC1
% % Second element represents the percentage of variance explained by the 2
% % first PCs (PC1 & PC2)
% % And so on...
% VarCumPercentage = cumsum(variance)/sum(variance)*100;
% figure('Color','w');
% title('Number of PCA Explaining 90% of the Total Variance of the Data');
% plot(VarCumPercentage);
% xlabel('Principal Components');
% ylabel('Cumulative Variance Explained [%]');
% box off;
% axis([0 648 0 100]);
% line([0 648], [90 90],'Color','r');
% line([44 44],[0 90],'Color','r','LineStyle','--');
% 
% % 44 features explain 90.1966 % of the variances

%% Cross-validation with PCA (a voir semaine pro)

% Partition: K = 10
cp_labels = cvpartition (labels,'kfold',10);

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
    
    % PCA 
    [coeff, score, variance] = pca(features, 'Centered', false); %A v?rifier centered data ou pas

   for j = 1:N
       
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


% ALice --> On peut enlever ca?
% size_labels = size(labels);
% N = size_labels(1); % N = total number of samples
% cp_labels = cvpartition (labels,'kfold',10);
% 
% % Initialization of error vectors
% errClassDiagQuadr = zeros(cp_labels.NumTestSets,1);
% 
% % For a different testSet i each time
% for i = 1:cp_labels.NumTestSets
%     
%     % Attention,ici le cp_N.taining rend les INDICES des train samples
%     % Quand trainIdx = 1 -> sample qui va dans le trainSet
%     trainIdx = cp_labels.training(i);
%     trainSet = features(trainIdx,:);
%     trainLabels = labels(trainIdx);
%    
%     % Attention, ici le cp_N.test rend les INDICES des test samples
%     % Quand testIdx = 1 -> sample va dans le testSet
%     testIdx = cp_labels.test(i);
%     testSet = features(testIdx,:);
%     testLabels = labels(testIdx);
%     
%     % Calculus of class errors
%     DiagQuadrclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagQuadratic');
%     DiagQuadr_y = predict(DiagQuadrclassifier,testSet);
%     errClassDiagQuadr(i) = classerrorOriginal(testLabels, DiagQuadr_y);
%     
% end
% 
% % % Mean of errors
% cvErrDiagQuadr = mean(errClassDiagQuadr);
% 
% % Standard deviations of errors
% cvStdDiagQuadr = std(errClassDiagQuadr);

%% Normalization of data (a voir semaine pro)
% features_normalized = zscore (features);
% [coeff_n, score_n, variance_n] = pca(features_normalized, 'Centered', false); %A v?rifier centered data ou pas
% 
% pca_normalized = zscore(score); %score ou features*coeff ?