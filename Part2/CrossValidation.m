close all;
clear all;
clc;

load('dataset_ERP.mat');

features = features(:,1:5:300);
size_labels = size(labels);

%% k-fold cross-validation
% Split data into k subsets of equal size
% Use k-1 subsets to train the classifier
% Remaining subset will be used for the testing --> for that make an
% iteration in order to one time use the 1 testing, and 2,3,4.. training,
% then the 2 testing, and 1,3,4,5... training etc...

N = size_labels(1); % N = total number of samples
NumTestSets = 10;
cp_N = cvpartition(N,'kfold',NumTestSets); % Outputs: Trainsize and Testsize contain the size (= number of samples) of each train/test set
cp_labels = cvpartition (labels,'kfold',NumTestSets);
C = cvpartition(N,'LeaveOut'); %"leave-one-out" cross validation

%train = cp.training(2); --> je ne comprends pas pk �a a une taille de
% 648...un subset est justement cens� avoir moins de samples que le nombre
% total de samples non ?
% train_size = size(cp.training(2));
% test= cp.test(2); --> pareil pour le test...
% test_size = size(cp.test(2));

%essai

for i = 1:NumTestSets
        
    test_set = cp_N.test(i);
        
        for j = 1:NumTestSets
            if (i == 10) & (j == i)
                train_set = cp_N.training(j);
            elseif (i ~= 10) & (j == i) 
                j = j + 1;
                train_set = cp_N.training(j);
            else
                train_set = cp_N.training(j);
            end
            %je ne sais pas vraiment comment faire pour faire un classifier
            %a chaque fois et le garder (l� � chaque boucle le classifier
            %est supprim�)
        Linclassifier = fitcdiscr(double(train_set),labels,'discrimtype', 'linear');
        Lin_y = predict(Linclassifier,double(test_set));
        [errorClassificationLin,errorClassLin] = classerror(labels, Lin_y);
        end
    
end