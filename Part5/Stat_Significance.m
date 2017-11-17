close all;
clear all;
clc;

load('dataset_ERP.mat');

%% Statistical Significance

N = 5; %number of mean_test_error_out that we have
test_error_out = []; %Outer fold's test error values (load from workspace)
mean_test_error_out = [0.2013 0.2103 0.2011 0.2450 0.1691]; %Mean test errors across outer folds (found before)
random_level = 0.5; %Random level of 50% (2-class problem)
alpha = 0.05;
significant = zeros(1,N);

for i = 1:N

standard_error = (test_error_out - mean_test_error_out(i))/std(test_error_out);

%Boxplot to verify if the data are normally distributed
figure('Color','w');
boxplot(test_error_out);
title('Boxplot of Outer folds test error values')
box off;

%Kolmogorov_smirnov test to check if data are normally distributed
s = kstest(standard_error);

if s == 0 % null hypothesis not rejected -> come from standard normal distribution
     % h = 0 -> null hypothesis not rejected -> come from same distribution
[h,p] = ttest(mean_test_error_out(i), random_level);
% Determine if the p-value is significant
        if p <= alpha
            significant(i) = 1; % null hypothesis rejected --> significant
        elseif p > alpha
            significant(i) = 0.5; % null hypothesis not rejected --> not significant (0.5 to change from 0 if it did not pass by the t-test)
        end
elseif s == 1 % null hypothesis rejected -> does not come from standard normal distribution
    %Case of non-parametric statistical test --> Wilcoxon signed rank test
    [v, w] = signrank(standard_error, random_level);
end

end