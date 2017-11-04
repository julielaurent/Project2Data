clear all;
close all;

data = load('dataset_ERP.mat');

features = data.features;
labels = data.labels;


L = size(labels, 1);
K = 10; 

N = 60;
%k-fold partitioning 

cv_L = cvpartition(labels,'kfold',K); %we obtain Trainsize about 584 and Test 64
[orderedInd, orderedPower] = rankfeat(features, labels, 'fisher');

errTest  = zeros(N, K);
errTrain = zeros(N, K);

for m = 1:1:N
    for n = 1:1:K
        indTrain = cv_L.training(n); %retourne 1 si le sample appartient ? train.
        indTest = cv_L.test(n); %retourne 1 si le sample appartient ? test.
        
        setTrain= features(indTrain,:); %rempli seulement si l'indice est 1
        labTrain = labels(indTrain);
     
        tridx = find(indTrain); %retrun only the index of the train or test set
        teidx = find(indTest);
    
        %[orderedInd, orderedPower] = rankfeat(features(tridx,:), labels(tridx), 'fisher'); %class the most discriminative features 
        Diag_Lin_classifier = fitcdiscr(features(tridx,orderedInd(1:m)), labels(tridx),'discrimtype', 'diagLinear');
   
        ythat_diaglin_train = predict(Diag_Lin_classifier, features(tridx, orderedInd(1:m)));  
        ythat_diaglin_test  = predict(Diag_Lin_classifier, features(teidx, orderedInd(1:m)));  
        
        errtr(m,n) = classerror(labels(tridx), ythat_diaglin_train); %error de depending on the number flods and the number of the m first most discriminative features
        errte(m,n) = classerror(labels(teidx), ythat_diaglin_test);
    end
end
mean_errte = mean(errte,2);
mean_errtr = mean(errtr,2);

opti_nfeat = find(mean_errte==min(mean_errte));
OPTI_Nfeat = opti_nfeat(1)

% %% Test
% figure()
% plot(mean_errte,'b','LineWidth',5)
% title('Test error depending on the number of features, fisher ranking', 'FontSize',15)
% hold on
% plot(errte); 
% legend('avg','1','2','3','4','5','6','7','8','9','10')
% xlabel('Number of Features', 'FontSize',13)
% ylabel('Mean Test Error', 'FontSize',13)
% 
% %%Training
% 
% figure()
% plot(mean_errtr,'b','LineWidth',5)
% title('Training error depending on the number of features, fisher ranking', 'FontSize',15)
% hold on
% plot(errtr); 
% legend('avg','1','2','3','4','5','6','7','8','9','10')
% xlabel('Number of Features',  'FontSize',13)
% ylabel('Mean Training Error' , 'FontSize',13)
