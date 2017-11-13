%% Nested cross-validation with forward feature selection and classifier selection

size_labels = size(labels);

% Total number of samples
N = size_labels(1);

Nfeature = 60; % number of features tried
nModel = 0; 
N = Nfeature * 3; %total number of models
Kout = 5; %number of outer loop folds
Kin = 10; %number of inner folds

% Options
opt = statset('Display','iter','MaxIter',100);

classifierType = {'diaglinear','linear','diagquadratic'};

model = struct('classifier',[],'number_of_features',[]);

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
    
    % Attention,ici le cp_N.training rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx_out = cp_labels_out.training(p);
    testIdx_out = cp_labels_out.test(p);
    trainLabels_out = labels(trainIdx_out);
    testLabels_out = labels(testIdx_out);
    
    
    %Choice of classifier
    for type = 1:3
        c = char(classifierType(type));
        
        % Inner partition on the train set of our outer-fold --> DEDANS ou
        % DEHORS???
        cp_labels_in = cvpartition (trainLabels_out,'kfold',Kin);
        
         % defines the criterion used to select features and to determine when to stop
            % si j'ai bien compris,x -> set, y -> labels, t -> test/validation, T -> train
         fun = @(xT,yT,xt,yt) length(yt)*(classerror(yt,predict(fitcdiscr(xT,yT,'discrimtype',c),xt)));
        
        % sel is logical vector indicating which features are finally chosen
        % hst is a scalar strucure with the fiels Crit (vector containing criterion
        % values at each step) and In (logical matrix -> row indicates feature
        % selected at each step)
        % PERFORMS A 10-FOLD CV FOR EACH CANDIDATE FEATURE SUBSET
        [sel,hst] = sequentialfs(fun,trainSet_out,trainLabels_out,'cv',cp_labels_in,'options',opt,'keepout',[1:300,2000:2400]);
        % SHOULD WE DO A KEEPOUT ???

        opt_validationError(type,p) = hst.Crit(end);
        nb_selectedFeatures(type,p) = find(hst.Crit == opt_validationError(type,p));
        trainSet_selectedFeatures = trainSet(:,sel);
        testSet_selectedFeatures = testSet(:,sel);
        selectedFeatures = find(sel);
    
        % Calculus of class errors
        DiagLinclassifier = fitcdiscr(trainSet_selectedFeatures,trainLabels,'discrimtype', c);
        DiagLin_y = predict(DiagLinclassifier,testSet_selectedFeatures);
        errClassDiagLin(type,p) = classerror(testLabels, DiagLin_y);
    end
    
    
    % J'ai pas regard? ? partir d'ici ----------------------------------
    % Best number of features according to inner cross-validation
    mean_validationerror_in = mean(validationerr_in,2);
    optimal_validationerror_in(p) = min(mean_validationerror_in);
    mean_trainingerror_in = mean(errTrain_in,2);
    optimal_trainingerror_in(p) = min(mean_trainingerror_in);
    bestModelNumber = find(mean_validationerror_in == optimal_validationerror_in(p));
    bestModel_in(p) = bestModelNumber(1); % Si plusieurs min egaux, je choisis le premier

    % Extract best model data 
    bestModelClassifier = model(bestModel_in(p)).classifier; 
    
    % Construct our data matrix with the selected number of features on the
    % ranking done one the training set of the outer fold
    for j = 1:model(bestModel_in(p)).number_of_features
       features_model_out = [features_model_out, features(:,orderedIndout(j))];
    end
     
    % Select the train and test data for the outer fold
    trainSet_out = features_model_out(trainIdx_out,:); 
    testSet_out = features_model_out(testIdx_out,:);
       
    % Classifier construction
    classifier_out = fitcdiscr(trainSet_out,trainLabels_out,'discrimtype', bestModelClassifier);

    % Calculus of class error on test set -> testing error (1xKout)
    yTest_out = predict(classifier_out,testSet_out);
    errTest_out(p) = classerror(testLabels_out, yTest_out);
    
    % Calculus of class error on train set -> training error (1xKout)
    yTrain_out = predict(classifier_out,trainSet_out);
    errTrain_out(p) = classerror(trainLabels_out, yTrain_out);
    
end

%Calculus of best model characteristics
model(bestModel_in)