close all;
clear all;
clc;

j = 324;

load('dataset_ERP.mat');

features = features(:,1:5:300);

% Separate train and test set 50%/50%  -> voir code Julie
Trainfeatures = features(1:j,:);
Testfeatures = features((j+1):648,:);
Trainlabels = labels(1:j);
Testlabels = labels((j+1):648);

%% Pbm avec Singular matrix
% Singular matrix -> pas inversible car determinant nul. Matrice de
% covariance a une ligne ou colonne = 0, car on a des features corr?l?s ?
% 100% (-> d?truit colonne ou ligne). -> QDA pas adapt? pour nous car trop
% de redondance dans nos donn?es