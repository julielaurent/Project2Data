close all;
clear all;
clc;

load('dataset_ERP.mat');

features = features(:,1:5:300);
size_labels = size(labels);


%% Normal k-fold cross-validation
% % Split data into k subsets of equal size
% % Use k-1 subsets to train the classifier
% % Remaining subset will be used for the testing --> for that make an
% % iteration in order to one time use the 1 testing, and 2,3,4.. training,
% % then the 2 testing, and 1,3,4,5... training etc...
% 
N = size_labels(1); % N = total number of samples
cp_N = cvpartition(N,'kfold',10); % Outputs: Trainsize and Testsize contain the size (= number of samples) of each train/test set
cp_labels = cvpartition (labels,'kfold',10);

% Initialization of error vectors
errClassLin = zeros(cp_N.NumTestSets,1);
errClassDiagLin = zeros(cp_N.NumTestSets,1);
errClassDiagQuadr = zeros(cp_N.NumTestSets,1);

% For a different testSet i each time
for i = 1:cp_N.NumTestSets
    
    % A rajouter pour derniere question
    %cp_N = repartition(cp_N);
    
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx = cp_N.training(i);
    trainSet = features(trainIdx,:);
    trainLabels = labels(trainIdx);
   
    % Attention, ici le cp_N.test rend les INDICES des test samples
    % Quand testIdx = 1 -> sample va dans le testSet
    testIdx = cp_N.test(i);
    testSet = features(testIdx,:);
    testLabels = labels(testIdx);
    
    % Calculus of class errors
    Linclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'linear');
    Lin_y = predict(Linclassifier,testSet);
    errClassLin(i) = classerrorOriginal(testLabels, Lin_y);
    
    DiagLinclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagLinear');
    DiagLin_y = predict(DiagLinclassifier,testSet);
    errClassDiagLin(i) = classerrorOriginal(testLabels, DiagLin_y);
   
    DiagQuadrclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagQuadratic');
    DiagQuadr_y = predict(DiagQuadrclassifier,testSet);
    errClassDiagQuadr(i) = classerrorOriginal(testLabels, DiagQuadr_y);
    
end

% % Mean of errors
cvErrLin = mean(errClassLin);
cvErrDiagLin = mean(errClassDiagLin);
cvErrDiagQuadr = mean(errClassDiagQuadr);

% Standard deviations of errors
cvStdLin = std(errClassLin);
cvStdDiagLin = std(errClassDiagLin);
cvStdDiagQuadr = std(errClassDiagQuadr);

% Plot
figure('Color','w');
title('10-fold Cross Validation Error');
hold on;
barwitherr([cvStdDiagLin ; cvStdLin ; cvStdDiagQuadr], [cvErrDiagLin ; cvErrLin ; cvErrDiagQuadr]);
ylabel('Class Error');
xticks([1 2 3]);
xticklabels({'DiagLinear','Linear','DiagQuadratic'});
%axis([0.5 3.5 0 0.1]);
box off;
hold off;

% Leave One Out Cross Validation

C = cvpartition(N,'LeaveOut'); %"leave-one-out" cross validation
% Error Calculation
errClassLinC = zeros(C.NumTestSets,1);
errClassDiagLinC = zeros(C.NumTestSets,1);
errClassDiagQuadrC = zeros(C.NumTestSets,1);

for i = 1:C.NumTestSets
    
    % A rajouter pour derniere question
    %C = repartition(C);
    
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdxC = C.training(i);
    trainSetC = features(trainIdxC,:);
    trainLabelsC = labels(trainIdxC);
    
    % Attention, ici le cp_N.test rend les INDICES des test samples
    % Quand testIdx = 1 -> sample va dans le testSet
    testIdxC = C.test(i);
    testSetC = features(testIdxC,:);
    testLabelsC = labels(testIdxC);
    
    % Calculus of class errors
    LinclassifierC = fitcdiscr(trainSetC,trainLabelsC,'discrimtype', 'linear');
    Lin_yC = predict(LinclassifierC,testSetC);
    errClassLinC(i) = classerrorOriginal(testLabelsC, Lin_yC);
    
    DiagLinclassifierC = fitcdiscr(trainSetC,trainLabelsC,'discrimtype', 'diagLinear');
    DiagLin_yC = predict(DiagLinclassifierC,testSetC);
    errClassDiagLinC(i) = classerrorOriginal(testLabelsC, DiagLin_yC);
    
    DiagQuadrclassifierC = fitcdiscr(trainSetC,trainLabelsC,'discrimtype', 'diagQuadratic');
    DiagQuadr_yC = predict(DiagQuadrclassifierC,testSetC);
    errClassDiagQuadrC(i) = classerrorOriginal(testLabelsC, DiagQuadr_yC);
    
end

% Mean of errors 
cvErrLinC = mean(errClassLinC);
cvErrDiagLinC = mean(errClassDiagLinC);
cvErrDiagQuadrC = mean(errClassDiagQuadrC);

% Standard deviations of errors
cvStdLinC = std(errClassLinC);
cvStdDiagLinC = std(errClassDiagLinC);
cvStdDiagQuadrC = std(errClassDiagQuadrC);

% Plot
figure('Color','w');
title('Leave One Out Cross Validation Error');
hold on;
barwitherr([cvStdDiagLinC ; cvStdLinC ; cvStdDiagQuadrC], [cvErrDiagLinC ; cvErrLinC ; cvErrDiagQuadrC]);
ylabel('Class Error');
xticks([1 2 3]);
xticklabels({'DiagLinear','Linear','DiagQuadratic'});
%axis([0.5 3.5 0 0.1]);
box off;
hold off;

%% PARTITION JULY

N = size_labels(1); % N = total number of samples
cp_N = cvpartition(N,'kfold',10); % Outputs: Trainsize and Testsize contain the size (= number of samples) of each train/test set
cp_labels = cvpartition (labels,'kfold',10);

% Initialization of error vectors
errClassDiagQuadr = zeros(cp_N.NumTestSets,1);

% For a different testSet i each time
for i = 1:cp_N.NumTestSets
    
    % A rajouter pour derniere question
    cp_N = repartition(cp_N);
    
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdx = cp_N.training(i);
    trainSet = features(trainIdx,:);
    trainLabels = labels(trainIdx);
   
    % Attention, ici le cp_N.test rend les INDICES des test samples
    % Quand testIdx = 1 -> sample va dans le testSet
    testIdx = cp_N.test(i);
    testSet = features(testIdx,:);
    testLabels = labels(testIdx);
    
    % Calculus of class errors
    DiagQuadrclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'diagQuadratic');
    DiagQuadr_y = predict(DiagQuadrclassifier,testSet);
    errClassDiagQuadr(i) = classerrorOriginal(testLabels, DiagQuadr_y);
    
end

% % Mean of errors
cvErrDiagQuadrpart = mean(errClassDiagQuadr);

% Standard deviations of errors
cvStdDiagQuadrpart = std(errClassDiagQuadr);

% Plot
figure('Color','w');
title('10-fold Cross Validation Error');
hold on;
barwitherr([cvStdDiagQuadr,cvStdDiagQuadrpart], [cvErrDiagQuadr,cvErrDiagQuadrpart]);
ylabel('Class Error');
xticks([1 2]);
xticklabels({'DiagQuadratic', 'DiagQuadratic with changing partition'});
%axis([0.5 3.5 0 0.1]);
box off;
hold off;

% Leave One Out Cross Validation

C = cvpartition(N,'LeaveOut'); %"leave-one-out" cross validation
% Error Calculation
errClassDiagQuadrC = zeros(C.NumTestSets,1);

for i = 1:C.NumTestSets
    
    % A rajouter pour derniere question
    C = repartition(C);
    
    % Attention,ici le cp_N.taining rend les INDICES des train samples
    % Quand trainIdx = 1 -> sample qui va dans le trainSet
    trainIdxC = C.training(i);
    trainSetC = features(trainIdxC,:);
    trainLabelsC = labels(trainIdxC);
    
    % Attention, ici le cp_N.test rend les INDICES des test samples
    % Quand testIdx = 1 -> sample va dans le testSet
    testIdxC = C.test(i);
    testSetC = features(testIdxC,:);
    testLabelsC = labels(testIdxC);
    
    % Calculus of class errors
    DiagQuadrclassifierC = fitcdiscr(trainSetC,trainLabelsC,'discrimtype', 'diagQuadratic');
    DiagQuadr_yC = predict(DiagQuadrclassifierC,testSetC);
    errClassDiagQuadrC(i) = classerrorOriginal(testLabelsC, DiagQuadr_yC);
    
end

% Mean of errors
cvErrDiagQuadrCpart = mean(errClassDiagQuadrC);

% Standard deviations of errors
cvStdDiagQuadrCpart = std(errClassDiagQuadrC);

% Plot
figure('Color','w');
title('Leave One Out Cross Validation Error');
hold on;
barwitherr([cvStdDiagQuadrC,cvStdDiagQuadrCpart], [cvErrDiagQuadrC,cvErrDiagQuadrCpart]);
ylabel('Class Error');
xticks([1 2]);
xticklabels({'DiagQuadratic','DiagQuadratic with changing partition'});
%axis([0.5 3.5 0 0.1]);
box off;
hold off;

%% Peut etre faire un graph avec normal cv ET repartition ? chaque boucle ?
% Jouer avec les hold on pour pas avoir ? tout refaire ??
% Commenter leave one out part -> enlever hold off ?la fin du 1er graph et
% mettre hold on ? la place -> faire tourner le code -> d?commenter la
% repartition -> refaire tourner le code

% Alice: J'ai fais comme ca, je sais pas si ca va. J'ai fais que avec
% Linear parce que long. Faire ttest???

% N = size_labels(1); % N = total number of samples
% cp_N = cvpartition(N,'kfold',10); % Outputs: Trainsize and Testsize contain the size (= number of samples) of each train/test set
% 
% for j=1:4
%     cp_N = repartition(cp_N);
%     % Initialization of error vectors
%     errClassLin = zeros(cp_N.NumTestSets,1);
% 
%     % For a different testSet i each time
%     for i = 1:cp_N.NumTestSets
%         % Attention,ici le cp_N.taining rend les INDICES des train samples
%         % Quand trainIdx = 1 -> sample qui va dans le trainSet
%         trainIdx = cp_N.training(i);
%         trainSet = features(trainIdx,:);
%         trainLabels = labels(trainIdx);
% 
%         % Attention, ici le cp_N.test rend les INDICES des test samples
%         % Quand testIdx = 1 -> sample va dans le testSet
%         testIdx = cp_N.test(i);
%         testSet = features(testIdx,:);
%         testLabels = labels(testIdx);
% 
%         % Calculus of class errors
%         Linclassifier = fitcdiscr(trainSet,trainLabels,'discrimtype', 'linear');
%         Lin_y = predict(Linclassifier,testSet);
%         errClassLin(i) = classerrorOriginal(testLabels, Lin_y);
%     end
%     
%     % % Mean of errors --> Alice: je pense que c'est plutot ca
%     % cvErrLin = sum(errClassLin)/sum(cp_N.TestSize);
%     cvErrLin(j) = mean(errClassLin);
% 
%     % Standard deviations of errors
%     cvStdLin(j) = std(errClassLin);
% end
% 
% % Plot
% figure('Color','w');
% title('10-fold Cross Validation Error comparison with changing of partition');
% hold on;
% barwitherr([cvStdLin(1) ; cvStdLin(2) ; cvStdLin(3) ; cvStdLin(4)], [cvErrLin(1) ; cvErrLin(2) ; cvErrLin(3) ; cvErrLin(4)]);
% ylabel('Class Error');
% xticks([1 2 3 4]);
% xticklabels({'Partition 1','Partition 2','Partition 3','Partition 4'});
% box off;
% hold off;
% 
% C = cvpartition(N,'LeaveOut'); %"leave-one-out" cross validation
% 
% for j=1:4
%     C = repartition(C);
% 
%     % Error Calculation
%     errClassLinC = zeros(C.NumTestSets,1);
% 
%     for i = 1:C.NumTestSets
%         % Attention,ici le cp_N.taining rend les INDICES des train samples
%         % Quand trainIdx = 1 -> sample qui va dans le trainSet
%         trainIdxC = C.training(i);
%         trainSetC = features(trainIdxC,:);
%         trainLabelsC = labels(trainIdxC);
% 
%         % Attention, ici le cp_N.test rend les INDICES des test samples
%         % Quand testIdx = 1 -> sample va dans le testSet
%         testIdxC = C.test(i);
%         testSetC = features(testIdxC,:);
%         testLabelsC = labels(testIdxC);
% 
%         % Calculus of class errors
%         LinclassifierC = fitcdiscr(trainSetC,trainLabelsC,'discrimtype', 'linear');
%         Lin_yC = predict(LinclassifierC,testSetC);
%         errClassLinC(i) = classerrorOriginal(testLabelsC, Lin_yC);
%     end
% 
%     % Mean of errors --> Alice: Idem
%     % cvErrLinC = sum(errClassLinC)/sum(C.TestSize);
%     cvErrLinC(j) = mean(errClassLinC);
% 
% 
%     % Standard deviations of errors
%     cvStdLinC(j) = std(errClassLinC);
% end
% 
% % Plot
% figure('Color','w');
% title('Leave-one out Cross Validation Error comparison with changing of partition');
% hold on;
% barwitherr([cvStdLinC(1) ; cvStdLinC(2) ; cvStdLinC(3) ; cvStdLinC(4)], [cvErrLinC(1) ; cvErrLinC(2) ; cvErrLinC(3) ; cvErrLinC(4)]);
% ylabel('Class Error');
% xticks([1 2 3 4]);
% xticklabels({'Partition 1','Partition 2','Partition 3','Partition 4'});
% box off;
% hold off;