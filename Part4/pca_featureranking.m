close all;
clear all;
clc;

load('dataset_ERP.mat');

%% PCA and feature ranking, with fixed classifier

N_pca = 60; % Number of PCs 
Kout = 10; %number of outer loop folds
Kin = 10; %number of inner folds

model = struct('nb_pc',[],'nb_rank',[]);

% Partition: K = 10
cp_labels_out = cvpartition (labels,'kfold',Kout);

% Initialization
 errTest_out = zeros(1,Kout); 
 errTrain_out = zeros(1,Kout); 
 validationerr_in  = [];  % Attention, not (N_pca*N_pca,Kout)!!!! Car pas 60*60 mod?les
 errTrain_in = [];   % Attention, not (N_pca*N_pca,Kout)!!!! Car pas 60*60 mod?les
 min_validationerror_in  = zeros(1,Kout); 
 min_trainingerror_in = zeros(1,Kout);
 bestModel_PCA  = zeros(1,Kout);
 bestModel_Rank = zeros(1,Kout); 

for p = 1:Kout
    trainSet_out_pca = [];
    testSet_out_pca = [];
    trainSet_out = [];
    testSet_out = [];
    
    % Attention,ici le cp_N.training rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx_out = cp_labels_out.training(p);
    testIdx_out = cp_labels_out.test(p);
    trainLabels_out = labels(trainIdx_out);
    testLabels_out = labels(testIdx_out);
    
     % Normalization and PCA
    [train_out, mu_out, sigma_out] = zscore(features(trainIdx_out,:));
    [coeff_out, score_out, variance_out] = pca(train_out);
    test_out = ((features(testIdx_out,:)' - mu_out')./sigma_out')'* coeff_out; % A comparer avec d'autres groupes
    
    % Inner partition on the train set of our outer-fold
    cp_labels_in = cvpartition (trainLabels_out,'kfold',Kin);
    
    % Kin fold
    for i = 1:Kin
        % Initialisation
        trainSet_in = [];
        testSet_in = [];
        nModel = 0;
        
        % Attention,ici le cp_N.taining rend les INDICES des train samples
         % Quand trainIdx = 1 -> sample qui va dans le trainSet
         trainIdx_in = cp_labels_in.training(i);
         trainLabels_in = trainLabels_out(trainIdx_in);
         testIdx_in = cp_labels_in.test(i);
         testLabels_in = trainLabels_out(testIdx_in);

        % PCA
        [train_in, mu_in, sigma_in] = zscore(features(trainIdx_in,:));
        [coeff_in, score_in, variance_in] = pca(train_in);
        test_in = ((features(testIdx_in,:)' - mu_out')./sigma_out')'* coeff_out; % A comparer avec d'autres groupes
        
       for j = 1:N_pca
           % Initialization
            trainSet_in_rank = [];
            testSet_in_rank = [];
           
            % Formation of train and test set of PCs
            trainSet_in = [trainSet_in, train_in(:,j)];
            testSet_in = [testSet_in, test_in(:,j)];

            % Rank of features for inner loop, on training set
            [orderedIndin, orderedPowerin] = rankfeat(trainSet_in,labels(trainIdx_in),'fisher');

            for k = 1:j 
                nModel = nModel + 1;
                model(nModel).nb_pc = j;
                model(nModel).nb_rank = k;
                
                % Formation of train and test set of PCs ordered by Fisher
                % ranking
                trainSet_in_rank = [trainSet_in_rank, trainSet_in(:,orderedIndin(k))];
                testSet_in_rank = [testSet_in_rank, testSet_in(:,orderedIndin(k))];

                % Classifier construction
                DiagLinclassifier_in = fitcdiscr(trainSet_in_rank,trainLabels_in,'discrimtype', 'diagLinear');

                % Calculus of class error on test set -> validation testing error (NxKin)
                yTest_in = predict(DiagLinclassifier_in,testSet_in_rank);
                validationerr_in(nModel,i) = classerror(testLabels_in, yTest_in);
          
                % Calculus of class error on train set -> training error (NxKin)
                yTrain_in = predict(DiagLinclassifier_in,trainSet_in_rank);
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
    bestModel_in(p) = bestModel(1);

    % Extract best model data 
    bestModel_PCA(p) = model(bestModel_in(p)).nb_pc; 
    bestModel_Rank(p) = model(bestModel_in(p)).nb_rank; 
    
    
    % Construct our data matrix with the selected number of features on the
    % ranking done one the training set of the outer fold, and the selected
    % number of PC
    for j = 1:bestModel_PCA(p)
       trainSet_out_pca  = [trainSet_out_pca, train_out(:,j)];
       testSet_out_pca  = [testSet_out_pca, test_out(:,j)];
    end
    
    % Rank of PCs for outer loop, on training set
    [orderedIndout, orderedPowerout] = rankfeat(trainSet_out_pca,labels(trainIdx_out),'fisher');
    
    for j = 1:bestModel_Rank(p)
       trainSet_out = [trainSet_out, trainSet_out_pca(:,orderedIndout(j))];
       testSet_out = [testSet_out, testSet_out_pca(:,orderedIndout(j))];
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
