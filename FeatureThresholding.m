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
    misClassA = 0;
    misClassB = 0;
    it = it + 1;
    
    labelf681 = (sf681 < tf681); % element is 1 if Class A
    
    for i = 1:648
        if (labels(i) == labelf681(i))
            nCorrect681 = nCorrect681 + 1;

        else
            if (labels(i) == 1)
                misClassA = misClassA + 1;
            else
                misClassB = misClassB + 1;
            end
        end
    end
    
errorClassification681(it) = 1 - nCorrect681/648;
errorClass681(it) = 0.5*misClassA/516 + 0.5*misClassB/132;

end

figure('Color','w');
th = -6:0.09:6;
plot(th,errorClassification681, th, errorClass681);
xlabel('Threshold'); ylabel('Error');
legend('Classification Error','Class Error');
box off;

optimalTH1 = th(find(errorClassification681 == min(errorClassification681)));
optimalTH2 = th(find(errorClass681 == min(errorClass681)));

%% Scatter
% We keep feature 362
figure('Color','w');
scatter(x(:,681),x(:,531),'.b');
hold on;
scatter(y(:,681),y(:,531),'.r');
xlabel('Feature 681'); ylabel('Feature 531');
hold on;
line([optimalTH1 optimalTH1],[-8 8],'Color','k','LineStyle',':');
hold on;
line([optimalTH2 optimalTH2],[-8 8],'Color','k','LineStyle','--');
legend('Samples from Class A (correct)','Samples from Class B (erroneous)','Threshold with minimal classification error','Threshold with minimal class error','Location','best');
hold off;

%% Scale to two features

errorClassification = zeros(121,1);
errorClass = zeros(121,1);
labelsDouble = [labels;labels];

sf = [features(:,681);features(:,531)];

it = 0;
for tf = -6:0.1:6
    
    nCorrect = 0;
    misClassA = 0;
    misClassB = 0;
    it = it + 1;
    
    labelf = (sf < tf); % element is 1 if Class A
    
    for i = 1:2*648
        if (labelsDouble(i) == labelf(i))
            nCorrect = nCorrect + 1;

        else
            if (labelsDouble(i) == 1)
                misClassA = misClassA + 1;
            else
                misClassB = misClassB + 1;
            end
        end
    end
    
errorClassification(it) = 1 - nCorrect/(2*648);
errorClass(it) = 0.5*misClassA/(2*516) + 0.5*misClassB/(2*132);

end

figure('Color','w');
th = -6:0.1:6;
plot(th,errorClassification, th, errorClass);
xlabel('Threshold'); ylabel('Error');
legend('Classification Error','Class Error');
box off;

optimalTH1Double = th(find(errorClassification == min(errorClassification)));
optimalTH2Double = th(find(errorClass == min(errorClass)));

figure('Color','w');
scatter(x(:,681),x(:,531),'.b');
hold on;
scatter(y(:,681),y(:,531),'.r');
xlabel('Feature 681'); ylabel('Feature 531');
hold on;
line([optimalTH1 optimalTH1],[-8 8],'Color','k','LineStyle',':');
hold on;
line([optimalTH2 optimalTH2],[-8 8],'Color','k','LineStyle','--');
legend('Samples from Class A (correct)','Samples from Class B (erroneous)','Threshold with minimal classification error','Threshold with minimal class error','Location','best');
hold off;
