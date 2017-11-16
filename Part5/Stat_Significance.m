close all;
clear all;
clc;

load('dataset_ERP.mat');

%% Statistical Significance

N = ; %number of mean_test_error_out that we have
test_error_out =; %Outer fold's test error values (load from workspace)
mean_test_error_out = []; %Mean test errors across outer folds (found before)
random_level = 0.5; %Random level of 50% (2-class problem)
alpha = 0.05;
significant = [1,N];

for i:N
    
% h = 0 -> null hypothesis not rejected -> come from same distribution
[h,p] = ttest(mean_test_error_out(i), random_level);

% Determine if the p-value is significant
if p <= alpha
    significant(i) = 1; %means that it is significant --> the null hypothesis is rejected
elseif 
    significant(i) = 0; %-->the null hypothesis is not rejected --> not significant
end

figure('Color','w');
boxplot(test_error_out);
title('Boxplot of Outer folds test error values')
box off;

end