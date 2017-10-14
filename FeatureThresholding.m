close all;
clear all;
clc;

load('dataset_ERP.mat');

%Class A: label '1', correct movement of cursor
%Class B: label '0', erroneous movement of cursor
x = features(find(labels == 1), :); %vector of class A
y = features(find(labels == 0), :); %vector of class B

%% Threshold choice with histograms ???
% % We take good features : 362, 366
% % Histograms can help us find the TH how ?????
% figure('Color','w');
% histogram(x(:,362));
% hold on;
% histogram(y(:,362));
% box off;
% hold off;
% figure('Color','w');
% histogram(x(:,366));
% hold on;
% histogram(y(:,366));
% box off;
% hold off;

%% Scatter
figure('Color','w');
%tf362 = min;
%tf366 = ;
scatter(features(:,362),features(:,366),'.');
xlabel('Feature 362'); ylabel('Feature 366');
hold on;
%ine(tf362);
hold off;

%% Classification
% sf362 = features(:,362);
% % tf362 = ;
% sf366 = features(:,366);
% % tf366 = ;
% labelf362 = (sf362 < tf362); % element is 1 if ClassA
% labelf366 = (sf366 < tf366); % element is 1 if ClassA
% 
% %% Error
% nCorrect362 = 0;
% nCorrect366 = 0;
% for i = 1:size(labels)
%     if (labels(i) == labelf362(i))
%         nCorrect362 = nCorrect362 + 1;
%     end
%     if (labels(i) == labelf366(i))
%         nCorrect362 = nCorrect366 + 1;
%     end
% end
% errorClassification362 = 1 - nCorrect362/size(labels);
% errorClassification366 = 1 - nCorrect366/size(labels);
% errorClass = ;
% 
% %% Plot error classification
% th = -4:0.5:6;
