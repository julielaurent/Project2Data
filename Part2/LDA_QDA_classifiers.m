close all;
clear all;
clc;

load('dataset_ERP.mat');
features = features(:,1:5:300);

%% Search for the model/classifier
Linclassifier = fitcdiscr(features,labels,'discrimtype', 'linear'); % LDA
DiagLinclassifier = fitcdiscr(features,labels,'discrimtype', 'diaglinear'); % LDA
DiagQuadrclassifier = fitcdiscr(features,labels,'discrimtype', 'diagquadratic'); % QDA
Quadrclassifier = fitcdiscr(features,labels,'discrimtype', 'quadratic'); %QDA

Lin_y = predict(Linclassifier,features);
DiagLin_y = predict(DiagLinclassifier,features);
DiagQuadr_y = predict(DiagQuadrclassifier,features);
Quadr_y = predict(Quadrclassifier,features);

%% Classification error and Class error
[errorClassificationLin,errorClassLin] = classerror(labels, Lin_y); % For Linear classifier
[errorClassificationDiagLin,errorClassDiagLin] = classerror(labels, DiagLin_y); % For Diagonal Linear classifier
[errorClassificationDiagQuadr,errorClassDiagQuadr] = classerror(labels, DiagQuadr_y); % For Diagonal Quadratic classifier
[errorClassificationQuadr,errorClassQuadr] = classerror(labels, Quadr_y); % For Quadratic classifier

% Plot of errors
%errorClassification = [errorClassificationLin,errorClassificationDiagLin,errorClassificationDiagQuadr,errorClassificationQuadr]
%errorClass = [errorClassLin,errorClassDiagLin,errorClassDiagQuadr,errorClassQuadr]

% figure('Color','w');
% title('Error for Each Classifier');
% hold on;
% y = [errorClassLin,errorClassificationLin;errorClassDiagLin,errorClassificationDiagLin;errorClassDiagQuadr,errorClassificationDiagQuadr;errorClassQuadr,errorClassificationQuadr];
% bar(y);
% ylabel('Error');
% xticks([1 2 3 4]);
% xticklabels({'Linear','DiagLinear','DiagQuadratic','Quadratic'});
% legend('Class error','Classification error');
% box off;
% hold off;
% --> Best error with quadratic classifier

figure('Color','w');
title('Class Error for Each Classifier');
hold on;
y = [errorClassLin;errorClassDiagLin;errorClassDiagQuadr;errorClassQuadr];
bar(y);
ylabel('Error');
xticks([1 2 3 4]);
xticklabels({'Linear','DiagLinear','DiagQuadratic','Quadratic'});
box off;
hold off;

%% Classifier with prior proba uniform
%  Quadratic has the best error but we look at all (d'apr?s assistant)
Linclassifierprior = fitcdiscr(features,labels,'discrimtype', 'linear', 'prior', 'uniform');
Linprior_y = predict(Linclassifierprior,features);
DiagLinclassifierprior = fitcdiscr(features,labels,'discrimtype', 'diaglinear', 'prior', 'uniform');
DiagLinprior_y = predict(DiagLinclassifierprior,features);
DiagQuadrclassifierprior = fitcdiscr(features,labels,'discrimtype', 'diagquadratic', 'prior', 'uniform');
DiagQuadrprior_y = predict(DiagQuadrclassifierprior,features);
Quadrclassifierprior = fitcdiscr(features,labels,'discrimtype', 'quadratic', 'prior', 'uniform');
Quadrprior_y = predict(Quadrclassifierprior,features);

% Classification and class error
[errorClassificationLinprior,errorClassLinprior] = classerror(labels, Linprior_y); 
[errorClassificationDiagLinprior,errorClassDiagLinprior] = classerror(labels, DiagLinprior_y); 
[errorClassificationDiagQuadrprior,errorClassDiagQuadrprior] = classerror(labels, DiagQuadrprior_y); 
[errorClassificationQuadrprior,errorClassQuadrprior] = classerror(labels, Quadrprior_y); 

% Plot of prior errors
%errorClassificationprior = [errorClassificationLinprior,errorClassificationDiagLinprior,errorClassificationDiagQuadrprior,errorClassificationQuadrprior]
%errorClassprior = [errorClassLinprior,errorClassDiagLinprior,errorClassDiagQuadrprior,errorClassQuadrprior]

figure('Color','w');
title('Error for Each Classifier with Uniform Prior');
hold on;
y = [errorClassLinprior,errorClassificationLinprior;errorClassDiagLinprior,errorClassificationDiagLinprior;errorClassDiagQuadrprior,errorClassificationDiagQuadrprior;errorClassQuadrprior,errorClassificationQuadrprior];
bar(y);
ylabel('Error');
xticks([1 2 3 4]);
xticklabels({'Linear','DiagLinear','DiagQuadratic','Quadratic'});
legend('Class error','Classification error');
box off;
hold off;

figure('Color','w');
title('Comparison of the class Error for different Prior Probabilities');
hold on;
y = [errorClassLin,errorClassLinprior;errorClassDiagLin,errorClassDiagLinprior;errorClassDiagQuadr,errorClassDiagQuadrprior;errorClassQuadr,errorClassQuadrprior];
bar(y);
ylabel('Error');
xticks([1 2 3 4]);
xticklabels({'Linear','DiagLinear','DiagQuadratic','Quadratic'});
legend('Empirical probability','Uniform probability');
box off;
hold off;