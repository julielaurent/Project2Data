close all;
clear all;
clc;

load('dataset_ERP.mat');

%% PCA and feature ranking, with fixed classifier

%N_rank = 60; % number of features tried
N_pca = 60; % Number of PCs 
Kout = 5; %number of outer loop folds
Kin = 10; %number of inner folds

% Partition: K = 10
cp_labels_out = cvpartition (labels,'kfold',Kout);

% Initialization
 errTest_out = zeros(1,Kout); 
 errTrain_out = zeros(1,Kout); 
 validationerr_in = zeros(Kin,N_pca,N_pca); 
 errTrain_in = zeros(Kin,N_pca,N_pca); 
 nbfeature_minTesterror_in = zeros(1,Kout); 
 min_validationerror_in  = zeros(1,Kout); 
 min_trainingerror_in  = zeros(1,Kout); 
 nb_pc_minTesterror_in = zeros(1,Kout); 
 min_validationerror_in_pca  = zeros(1,Kout); 
 min_trainingerror_in_pca  = zeros(1,Kout); 

for p = 1:Kout
    features_model_pca_out = [];
    features_model_rank_out = [];
    
    % Attention,ici le cp_N.training rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx_out = cp_labels_out.training(p);
    testIdx_out = cp_labels_out.test(p);
    trainLabels_out = labels(trainIdx_out);
    testLabels_out = labels(testIdx_out);
    
     % PCA --> Pas trop sure de moi la, si on fait PCA sur toutes les
    % donn?ess
    [coeff_out, score_out, variance_out] = pca(features, 'Centered', false); %A v?rifier centered data ou pas

    
    % Inner partition on the train set of our outer-fold
    cp_labels_in = cvpartition (trainLabels_out,'kfold',Kin);
    
    % Kin fold
    for i = 1:Kin
    
        % Initialisation
        features_model_pca_in = [];
        nModel = 0;
        
        % Attention,ici le cp_N.taining rend les INDICES des train samples
         % Quand trainIdx = 1 -> sample qui va dans le trainSet
         trainIdx_in = cp_labels_in.training(i);
         trainLabels_in = trainLabels_out(trainIdx_in);
         testIdx_in = cp_labels_in.test(i);
         testLabels_in = trainLabels_out(testIdx_in);

        % PCA
        [coeff_in, score_in, variance_in] = pca(features(trainIdx_out,:), 'Centered', false); %A v?rifier centered data ou pas

       for j = 1:N_pca
            features_model_rank_in = [];

            features_model_pca_in = [features_model_pca_in, score_in(:,j)];

            % Rank of features for inner loop, on training set: v?rifier si on laisse fisher
            [orderedIndin, orderedPowerin] = rankfeat(features_model_pca_in(trainIdx_in,:),labels(trainIdx_in),'fisher');

            for k = 1:j  % pas sure
                nModel = nModel + 1;
                features_model_rank_in = [features_model_rank_in, features_model_pca_in(:,orderedIndin(k))];

                % Construction of train and test set for inner loop
                trainSet_in = features_model_rank_in(trainIdx_in,:);
                testSet_in = features_model_rank_in(testIdx_in,:);

                % Classifier construction
                DiagLinclassifier_in = fitcdiscr(trainSet_in,trainLabels_in,'discrimtype', 'diagLinear');

                % Calculus of class error on test set -> validation testing error (NxKin)
                yTest_in = predict(DiagLinclassifier_in,testSet_in);
                validationerr_in(nModel,i) = classerror(testLabels_in, yTest_in);
          
                % Calculus of class error on train set -> training error (NxKin)
                yTrain_in = predict(DiagLinclassifier_in,trainSet_in);
                errTrain_in(nModel,i) = classerror(trainLabels_in, yTrain_in);
            end
       end      
    end
    
    % Best number of PCs and features according to inner cross-validation
    mean_validationerror_in = mean(validationerr_in,2);
    min_validationerror_in(p) = min(min(mean_validationerror_in)); 
    mean_trainingerror_in = mean(errTrain_in,2);
    min_trainingerror_in(p) = min(min(mean_trainingerror_in));
    bestModel = find(mean_validationerror_in == min_validationerror_in(p));
    bestModel(p) = bestModel(1);
    
    % Construct our data matrix with the selected number of features on the
    % ranking done one the training set of the outer fold, and the selected
    % number of PC
    for j = 1:bestModel_PCA(p)
       features_model_pca_out = [features_model_pca_out, score_out(:,j)];
    end
    
    % Rank of PCs for outer loop, on training set
    [orderedIndout, orderedPowerout] = rankfeat(features_model_pca_out(trainIdx_out,:),labels(trainIdx_out),'fisher');
    
    for j = 1:bestModel_Rank(p)
       features_model_rank_out = [features_model_rank_out, features_model_pca_out(:,orderedIndout(j))];
    end
   
     
    % Select the train and test data for the outer fold
    trainSet_out = features_model_rank_out(trainIdx_out,:); 
    testSet_out = features_model_rank_out(testIdx_out,:);
       
    % Classifier construction
    DiagLinclassifier_out = fitcdiscr(trainSet_out,trainLabels_out,'discrimtype', 'diagLinear');

    % Calculus of class error on test set -> testing error (1xKout)
    yTest_out = predict(DiagLinclassifier_out,testSet_out);
    errTest_out(p) = classerror(testLabels_out, yTest_out);
    
    % Calculus of class error on train set -> training error (1xKout)
    yTrain_out = predict(DiagLinclassifier_out,trainSet_out);
    errTrain_out(p) = classerror(trainLabels_out, yTrain_out);

end
