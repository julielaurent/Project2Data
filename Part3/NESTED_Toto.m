clear all;
close all;
clc

data = load('dataset_ERP.mat');

features = data.features;
labels = data.labels;


L = size(labels, 1);
Kout = 3; %outter folds
Kin = 4; %inner folds

Nfeat = 60;
%k-fold partitioning 

cv_outter = cvpartition(labels,'kfold',Kout); %partition of the outter folds
[orderedInd, orderedPower] = rankfeat(features, labels, 'fisher'); %feature ranking 

errTest_in  = zeros(Nfeat, Kin); %validation error, inner fold
errTrain_in = zeros(Nfeat, Kin); %training error, inner fold 
errTest_out = zeros(1, Kout); %testing error, outter fold

mean_errTest_in = zeros(Kin, 1);
mean_errTrain_in = zeros(Kin, 1);

for m = 1:Kout
    
    indTrain_out = cv_outter.training(m); %retourne 1 si le sample appartient ? train.
    indTest_out = cv_outter.test(m); %retourne 1 si le sample appartient ? test.
    setTrain_out = features(indTrain_out,:); %rempli seulement si l'indice est 1
    labTrain_out  = labels(indTrain_out);
    tridx_out = find(indTrain_out); %return only the index of the train or test set
    teidx_out = find(indTest_out);
    
    cv_inner = cvpartition(labTrain_out,'kfold',Kin);
    %partition de deux outter folds en 4 inner folds (3train, 1validation)
    
    for n = 1:Kin
        
        indTrain_in = cv_inner.training(n); %retourne 1 si le sample appartient ? train.
        indTest_in = cv_inner.test(n); %retourne 1 si le sample appartient ? test.
        setTrain= features(indTrain_in,:); %rempli seulement si l'indice est 1
        labTrain = labels(indTrain_in);
     
        tridx = find(indTrain_in); %return only the index of the train or test set
        teidx = find(indTest_in);
        
        for o = 1:Nfeat
    
            %[orderedInd, orderedPower] = rankfeat(features(tridx,:), labels(tridx), 'fisher'); %class the most discriminative features 
            Diag_Lin_classifier = fitcdiscr(features(tridx,orderedInd(1:o)), labels(tridx),'discrimtype', 'diagLinear');
   
            ythat_diaglin_train = predict(Diag_Lin_classifier, features(tridx, orderedInd(1:o)));  
            ythat_diaglin_test  = predict(Diag_Lin_classifier, features(teidx, orderedInd(1:o))); 
            
            errTrain_in(o,n) = classerror(labels(tridx), ythat_diaglin_train); %error de depending on the number flods and the number of the m first most discriminative features
            errTest_in(o,n) = classerror(labels(teidx), ythat_diaglin_test);
        end
        
    end
    
    
    mean_errTest_in = mean(errTest_in,2); %calcul de la validation error pour inner
    mean_errTrain_in = mean(errTrain_in,2); %calcul de la testing error pour inner 
  
    opti_nfeat = find(mean_errTest_in==min(mean_errTest_in));
    
    OPTI_Nfeat(m) = opti_nfeat(1);
    
    %opti_nfeat(1) returns the first minim if there is several same min.
    
    Diag_Lin_classifier = fitcdiscr(setTrain_out(1:OPTI_Nfeat(m),:), labTrain_out(1:OPTI_Nfeat(m),:),'discrimtype', 'diagLinear');
   
    ythat_train_out= predict(Diag_Lin_classifier, features(tridx_out, :));  
    ythat_test_out  = predict(Diag_Lin_classifier, features(teidx_out, :)); 
            
    %errtr(o,n) = classerror(labels(tridx_out), ythat_diaglin_train); %error de depending on the number flods and the number of the m first most discriminative features
    errTest_out(m) = classerror(labels(teidx_out), ythat_test_out);
    
end


YOUPI = errTest_out %Test error for each outter fold 
OPTI_Nfeat %Hyperparameter (optimal number of features)
