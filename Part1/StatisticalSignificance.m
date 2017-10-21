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

figure('Color','w');
subplot(1,2,1)
histogram(x(:,681));
hold on
histogram(y(:,681));
box off;
legend('Correct Cursor Movement','Erroneous Cursor Movement');
title('Histogram of feature 681')
hold off;
subplot(1,2,2)
histogram(x(:,305));
hold on
histogram(y(:,305));
box off;
title('Histogram of feature 305')
hold off;

%% Boxplots

% % Bad features
% % Similar feature distribution for Class A and B
% figure('Color','w');
% boxplot(features(:,302),labels);
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;
% title('Boxplot of feature 302')
% figure('Color','w');
% boxplot(features(:,303),labels);
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;
% title('Boxplot of feature 303')
% 
% % Good features
% % Different feature distribution for Class A and B
% figure('Color','w');
% boxplot(features(:,681),labels);
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;
% title('Boxplot of feature 681')
%figure('Color','w');
% boxplot(features(:,531),labels);
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;
% title('Boxplot of feature 531')


figure('Color','w');
subplot(1,2,1)
boxplot(features(:,681),labels);
xticklabels({'Erroneous','Correct'});
box off;
title('Boxplot of feature 681')
subplot(1,2,2)
boxplot(features(:,305),labels);
xticklabels({'Erroneous','Correct'});
box off;
title('Boxplot of feature 305')


%% Boxplots with Notch

% Bad features
% Similar feature distribution for Class A and B
% figure('Color','w');
% boxplot(features(:,302),labels,'Notch','on');
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;
% title('Notch boxplot of feature 302')
% figure('Color','w');
% boxplot(features(:,303),labels,'Notch','on');
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;
% title('Notch boxplot of feature 303')

% Good features
% Different feature distribution for Class A and B
% figure('Color','w');
% boxplot(features(:,681),labels,'Notch','on');
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% title('Notch boxplot of feature 681')
% box off;
% figure('Color','w');
% boxplot(features(:,531),labels,'Notch','on');
% xticklabels({'Erroneous Cursor Movement','Correct Cursor Movement'});
% box off;
% title('Notch boxplot of feature 531')

figure('Color','w');
subplot(1,2,1)
boxplot(features(:,681),labels,'Notch','on');
xticklabels({'Erroneous','Correct'});
title('Notch boxplot of feature 681')
box off;
subplot(1,2,2)
boxplot(features(:,305),labels,'Notch','on');
xticklabels({'Erroneous','Correct'});
box off;
title('Notch boxplot of feature 305')

%% T-tests

% Bad features = similar for each class
% h = 0 -> null hypothesis not rejected -> come from same distribution
[hBad1,pBad1] = ttest2(x(:,305),y(:,305));
[hBad2,pBad2] = ttest2(x(:,303),y(:,303));
% For both examples, the null hypothesis cannot be rejected at level of
% alpha -> classes not significantly different for those features
% p > 0.05 (about 0.7)

% Good features = different for each class
% h = 0 -> null hypothesis not rejected -> come from same distribution
[hGood1,pGood1] = ttest2(x(:,681),y(:,681));
[hGood2,pGood2] = ttest2(x(:,531),y(:,531));
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
    hX = lillietest(x(:,i));
    hY = lillietest(y(:,i));
    if (hX == 0 && hY == 0)
        [hGood(i),pGood(i)] = ttest2(x(:, i),y(:, i), 'Vartype','unequal'); % specify that our variances are not equal
    end
end
[corrected_p, h] = bonf_holm(pGood,alpha); % Bonferonni correction
% to avoid that values are equal to 0 : they would thus be minimals
zeroToReplaceIdx = find(corrected_p == 0);
corrected_p(zeroToReplaceIdx) = 1;

BestDiscrFeat = find(corrected_p == min(corrected_p))

% 4th best pvalue because further away from 681 -> 531
corrected_pSorted = sort(corrected_p);
SecondBestPvalue = corrected_pSorted(4);
SecondBestDisctFeat = find(corrected_p == SecondBestPvalue)