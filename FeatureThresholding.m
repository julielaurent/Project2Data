close all;
clear all;
clc;

load('dataset_ERP.mat');

%Class A: label '1', correct movement of cursor
%Class B: label '0', erroneous movement of cursor
x = features(find(labels == 1), :); %vector of class A
y = features(find(labels == 0), :); %vector of class B

%% Threshold choice with histograms ???
% % We take good features : 362, 378
% % Histograms can help us find the TH how ?????
% figure('Color','w');
% histogram(x(:,362));
% hold on;
% histogram(y(:,362));
% box off;
% hold off;
% figure('Color','w');
% histogram(x(:,378));
% hold on;
% histogram(y(:,362));
% box off;
% hold off;

%% Scatter
% figure('Color','w');
% tf362 = min;
% tf378 = ;
% scatter(features(:,362),features(:,378),'.');
% hold on;
% %ine(tf362);
% hold off;

%% Classification
sf362 = features(:,362);
% tf362 = ;
sf378 = features(:,378);
% tf378 = ;
labelf362 = (sf362 < tf362); % element is 1 if ClassA
labelf378 = (sf378 < tf378); % element is 1 if ClassA

%% Error
nCorrect362 = 0;
nCorrect378 = 0;
for i = 1:size(labels)
    if (labels(i) == labelf362(i))
        nCorrect362 = nCorrect362 + 1;
    end
    if (labels(i) == labelf378(i))
        nCorrect362 = nCorrect378 + 1;
    end
end
errorClassification362 = 1 - nCorrect362/size(labels);
errorClassification378 = 1 - nCorrect378/size(labels);
errorClass = ;

%% Plot error classification
th = -4:0.5:6;
