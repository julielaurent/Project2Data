close all;
clear all;
clc;

load('dataset_ERP.mat');

%Class A: label '1', correct movement of cursor
%Class B: label '0', erroneous movement of cursor
x = features(find(labels == 1), :); %vector of class A
y = features(find(labels == 0), :); %vector of class B

%% Histograms

% for k=300:400
%     figure('Name',num2str(k),'color','w');
% 
%     histogram(x(:,k));
%     hold on
%     histogram(y(:,k));
%     box off;
%     legend('Correct Cursor Movement','Erroneous Cursor Movement');
%     hold off;
%     j = waitforbuttonpress;
% end
% Examples of similar feature distribution for both class : 300, 306, 302,
% 303, 311
% Examples of very different feature distribution for each class : 345, 346,
% 359,360, 362, 366, 367, 368, 369, 378

%% Boxplots

% % % Bad features
% % % Similar feature distribution for Class A and B
% figure('Color','w');
% boxplot(features(:,302),labels);
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;
% figure('Color','w');
% boxplot(features(:,303),labels);
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;
% % 
% % % Good features
% % % Different feature distribution for Class A and B
% figure('Color','w');
% boxplot(features(:,362),labels);
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;
% figure('Color','w');
% boxplot(features(:,366),labels);
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;


%% Boxplots with Notch

% Bad features
% Similar feature distribution for Class A and B
figure('Color','w');
boxplot(features(:,302),labels,'Notch','on');
xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
box off;
figure('Color','w');
boxplot(features(:,303),labels,'Notch','on');
xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
box off;

% Good features
% Different feature distribution for Class A and B
figure('Color','w');
boxplot(features(:,362),labels,'Notch','on');
xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
box off;
figure('Color','w');
boxplot(features(:,366),labels,'Notch','on');
xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
box off;

%% T-tests

% Bad features = similar for each class
% h = 0 -> null hypothesis not rejected -> come from same distribution
[hBad1,pBad1] = ttest2(x(:,302),y(:,302));
[hBad2,pBad2] = ttest2(x(:,303),y(:,303));
% For both examples, the null hypothesis cannot be rejected at level of
% alpha -> classes not significantly different for those features
% p > 0.05 (about 0.7)

% Good features = different for each class
% h = 0 -> null hypothesis not rejected -> come from same distribution
[hGood1,pGood1] = ttest2(x(:,362),y(:,362));
[hGood2,pGood2] = ttest2(x(:,366),y(:,366));
% For both examples, the null hypothesis is rejected at level of alpha
% p < 0.05 (negative exposents) -> classes significantly different for those
% features


% Ttest cannot be used for all features because not all features are
% normally distributed. Bad features might be nirmally distributed butwe
% should check. Good features have 2 different distributions for each class
% so we cannot use the t-tes
% 
% If we apply ttest to all features separately and we find one that is
% significant, it does not mean that the distributions (for this feature)
% of the two classes are different. Multiple testing requires a correction
% because the more population you test, the more chance you have to get a
% positive. If features are normally distributed, we could use the
% Bonferroni correction for multiple testing.

% To find the "best" feature
alpha = 0.05;
for i=1:2400  
    [hGood(i),pGood(i)] = ttest2(x(:, i),y(:, i), 'Vartype','unequal'); % specify that our variances are not equal
    [corrected_p(i), h(i)] = bonf_holm(pGood(i),alpha); % Bonferonni correction
end

BestDiscrFeat = find(corrected_p == min(corrected_p))