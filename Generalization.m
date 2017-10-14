close all;
clear all;
clc;

load('dataset_ERP.mat');

%% Split Dataset
trainSet1 = features(1:430,:);
testSet2 = features(431:648,:);
labelsTrain = labels(1:430);
labelsTest = labels(431:648);
ratioOrigin = 516/136;

%Class A: label '1', correct movement of cursor
%Class B: label '0', erroneous movement of cursor
x1 = trainSet1(find(labelsTrain == 1), :); %vector of class A
y1 = trainSet1(find(labelsTrain == 0), :); %vector of class B
sizeX1 = size(x1);
sizeY1 = size(y1);
ratioTrain = sizeX1(1)/sizeY1(1);

x2 = testSet2(find(labelsTest == 1), :); %vector of class A
y2 = testSet2(find(labelsTest == 0), :); %vector of class B
sizeX2 = size(x2);
sizeY2 = size(y2);
ratioTest = sizeX2(1)/sizeY2(1);

%% Classification and optimal TH with TRAINING

errorClassification362 = zeros(134,1);
errorClass362 = zeros(134,1);

sf362 = trainSet1(:,362);
it = 0;
for tf362 = -6:0.09:6
    
    nCorrect362 = 0;
    misClassA = 0;
    misClassB = 0;
    it = it + 1;
    
    labelf362 = (sf362 > tf362); % element is 1 if Class A
    
    for i = 1:430
        if (labelsTrain(i) == labelf362(i))
            nCorrect362 = nCorrect362 + 1;

        else
            if (labelsTrain(i) == 1)
                misClassA = misClassA + 1;
            else
                misClassB = misClassB + 1;
            end
        end
    end
    
errorClassification362(it) = 1 - nCorrect362/(sizeX1(1)+sizeY1(1));
errorClass362(it) = 0.5*misClassA/sizeX1(1) + 0.5*misClassB/sizeY1(1);

end

figure('Color','w');
th = -6:0.09:6;
plot(th,errorClassification362, th, errorClass362);
xlabel('Threshold'); ylabel('Error');
legend('Classification Error','Class Error');
box off;

optimalTH1 = th(find(errorClassification362 == min(errorClassification362)));
optimalTH2 = th(find(errorClass362 == min(errorClass362)));

errorClassification362Train = errorClassification362(find(optimalTH1));
errorClass362Train = errorClass362(find(optimalTH2));

% Scatter
% We keep feature 362
figure('Color','w');
scatter(x1(:,362),x1(:,366),'.b');
hold on;
scatter(y1(:,362),y1(:,366),'.r');
xlabel('Feature 362'); ylabel('Feature 366');
hold on;
vline(optimalTH1,'k--');
hold on;
vline(optimalTH2,'k-.');
legend('Samples from Class A','Samples from Class B');
hold off;

%% Classification of TESTING set with found TH

sf362 = testSet2(:,362);
    
nCorrect362 = 0;
misClassA = 0;
misClassB = 0;

labelf362 = (sf362 > optimalTH2); % element is 1 if Class A

% Performance of Test
S2 = size(testSet2);
for i = 1:S2(1)
    if (labelsTest(i) == labelf362(i))
        nCorrect362 = nCorrect362 + 1;

    else
        if (labelsTest(i) == 1)
            misClassA = misClassA + 1;
        else
            misClassB = misClassB + 1;
        end
    end
end
    
errorClassification362Test = 1 - nCorrect362/(sizeX2(1)+sizeY2(1));
errorClass362Test = 0.5*misClassA/sizeX2(1) + 0.5*misClassB/sizeY2(1);


figure('Color','w');
th = -6:0.09:6;
plot(th,errorClassification362, th, errorClass362);
xlabel('Threshold'); ylabel('Error');
legend('Classification Error','Class Error');
box off;

% Scatter
% We keep feature 362
figure('Color','w');
scatter(x1(:,362),x1(:,366),'.b');
hold on;
scatter(y1(:,362),y1(:,366),'.r');
xlabel('Feature 362'); ylabel('Feature 366');
hold on;
vline(optimalTH1,'k--');
hold on;
vline(optimalTH2,'k-.');
legend('Samples from Class A','Samples from Class B');
hold off;
