close all;
clear all;
clc;

load('dataset_ERP.mat');

%Class A: label '1', correct movement of cursor
%Class B: label '0', erroneous movement of cursor
x = features(find(labels == 1), :); %vector of class A
y = features(find(labels == 0), :); %vector of class B

%% Threshold choice with histograms 
% % We take good features : 681 and 531
figure('Color','w');
title('Feature 681');
hold on;
histogram(x(:,681));
hold on;
histogram(y(:,681));
legend('Class A : correct movement of cursor','Class B : erroneous movement of cursor');
box off;
hold off;

figure('Color','w');
title('Feature 531');
hold on;
histogram(x(:,531));
hold on;
histogram(y(:,531));
legend('Class A : correct movement of cursor','Class B : erroneous movement of cursor');
box off;
hold off;

%% Classification and optimal TH
errorClassification681 = zeros(134,1);
errorClass681 = zeros(134,1);

sf681 = features(:,681);
it = 0;

for tf681 = -6:0.09:6
    
    nCorrect681 = 0;
    it = it + 1;
    
    labelsOutput = (sf681 < tf681); % element is 1 if Class A
    errorClass681(it) = classerrorOriginal(labels, labelsOutput);
    
    for i = 1:648
        if (labels(i) == labelsOutput(i))
            nCorrect681 = nCorrect681 + 1;
        end
    end
    
    errorClassification681(it) = 1 - nCorrect681/648;

end

figure('Color','w');
title('Errors for Feature 681');
hold on;
th = -6:0.09:6;
plot(th, errorClassification681, th, errorClass681);
xlabel('Threshold'); ylabel('Error');
legend('Classification Error','Class Error');
box off;

optimalTH = th(find(errorClass681 == min(errorClass681)));

%% Scatter
figure('Color','w');
title('Feature 681 vs. Feature 531');
hold on;
scatter(x(:,681),x(:,531),'.b');
hold on;
scatter(y(:,681),y(:,531),'.r');
xlabel('Feature 681'); ylabel('Feature 531');
hold on;
line([optimalTH optimalTH],[-8 8],'Color','k','LineStyle','--');
legend('Samples from Class A (correct cursor movement)','Samples from Class B (erroneous cursor movement)','Threshold classifier with minimal class error','Location','best');
hold off;

