close all;
clear all;
clc;

load('dataset_ERP.mat');

%% Statistical Significance

N = 5; %number of mean_test_error_out that we have
test_error_out = [0.1652 0.2213 0.0588 0.2308 0.0577 0.2692 0.1538 0.3365 0.2115 0.3084; 0.1938 0.2633 0.2409 0.1538 0.2115 0.2692 0.2019 0.2404 0.1058 0.2225; 0.3273 0.2115 0.2990 0.1635 0.2115 0.1923 0.1923 0.2500 0.0673 0.0965; 0.2021 0.2857 0.2500 0.2692 0.1635 0.2885 0.2692 0.1923 0.3462 0.1833; 0.1833 0.1758 0.0812 0.2596 0.1442 0.1538 0.2019 0.1442 0.1538 0.1931]; %Outer fold's test error values
models = {'Fisher Ranking', 'Fisher Ranking + Classifier', 'FFS', 'PCA + Fisher Ranking', 'FFS + Classifier'};

% Feature selection (Fisher ranking) = 0.2013 --> 1rst line of
% test-error-out (in loop for --> i = 1)
% Feature selection (Feature ranking) + Classifier choice = 0.2103 --> 2nd
% line of test_error_out (in loop for --> i = 2)
% Feature Selection (FFS) = 0.2011 --> 3rd line of test-error-out (in loop for --> i = 3)
% Feature Engineering (PCA) + Feature Selection (Fisher ranking) = 0.2450
% --> 4th line of test-error-out (in loop for --> i = 4)
% Feature Selection (FFS) + Classifier choice = 0.1691 --> 5th line of
% test-error-out (in loop for --> i = 5)

mean_test_error_out = [0.2013 0.2103 0.2011 0.2450 0.1691]; %Mean test errors across outer folds
random_level = 0.5; %Random level of 50% (2-class problem)
alpha = 0.05;
significant = zeros(1,N);

figure('Color','w');
%title('Test Error Distribution across Outer Folds');
hold on;

for i = 1:N

standardized_error = (test_error_out(i,:) - mean_test_error_out(i))/std(test_error_out(i,:));

%Histogram to verify if the data are normally distributed
subplot(3,2,i);
histogram(test_error_out(i,:));
xlabel('Error Value');
title(char(models(i)));
box off;
axis([0 0.5 0 8])

%Kolmogorov_smirnov test to check if data are normally distributed
s = kstest(standardized_error);

if s == 0 % null hypothesis not rejected -> come from standard normal distribution
     % h = 0 -> null hypothesis not rejected -> come from same distribution
[h,p] = ttest(test_error_out(i,:), random_level);
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