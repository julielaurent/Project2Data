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
figure('Color','w');
histogram(x(:,362));
hold on;
histogram(y(:,362));
legend('Correct movement of Cursor','Erroneous Movement of Cursor');
box off;
hold off;
% figure('Color','w');
% histogram(x(:,366));
% hold on;
% histogram(y(:,366));
% box off;
% hold off;

%% Classification and optimal TH
errorClassification362 = zeros(134,1);
errorClass362 = zeros(134,1);

sf362 = features(:,362);
it = 0;
for tf362 = -6:0.09:6
    
    nCorrect362 = 0;
    misClassA = 0;
    misClassB = 0;
    it = it + 1;
    
    labelf362 = (sf362 > tf362); % element is 1 if Class A
    
    for i = 1:648
        if (labels(i) == labelf362(i))
            nCorrect362 = nCorrect362 + 1;

        else
            if (labels(i) == 1)
                misClassA = misClassA + 1;
            else
                misClassB = misClassB + 1;
            end
        end
    end
    
errorClassification362(it) = 1 - nCorrect362/648;
errorClass362(it) = 0.5*misClassA/516 + 0.5*misClassB/132;

end

figure('Color','w');
th = -6:0.09:6;
plot(th,errorClassification362, th, errorClass362);
xlabel('Threshold'); ylabel('Error');
legend('Classification Error','Class Error');
box off;

optimalTH1 = th(find(errorClassification362 == min(errorClassification362)));
optimalTH2 = th(find(errorClass362 == min(errorClass362)));

%% Scatter
% We keep feature 362
figure('Color','w');
scatter(x(:,362),x(:,366),'.b');
hold on;
scatter(y(:,362),y(:,366),'.r');
xlabel('Feature 362'); ylabel('Feature 366');
hold on;
vline(optimalTH1,'k--');
hold on;
vline(optimalTH2,'k-.');
legend('Samples from Class A','Samples from Class B');
hold off;

%% Scale to two features

errorClassification = zeros(121,1);
errorClass = zeros(121,1);
labelsDouble = [labels;labels];

sf = [features(:,362);features(:,366)];

it = 0;
for tf = -6:0.1:6
    
    nCorrect = 0;
    misClassA = 0;
    misClassB = 0;
    it = it + 1;
    
    labelf = (sf > tf); % element is 1 if Class A
    
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
scatter(x(:,362),x(:,366),'.b');
hold on;
scatter(y(:,362),y(:,366),'.r');
xlabel('Feature 362'); ylabel('Feature 366');
hold on;
vline(optimalTH1Double,'k--');
hold on;
vline(optimalTH2Double,'k-.');
legend('Samples from Class A','Samples from Class B');
hold off;
