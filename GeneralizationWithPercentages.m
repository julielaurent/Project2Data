close all;
clear all;
clc;

load('dataset_ERP.mat');
ratioOrigin = 516/136;
iPercentage = 0;
errorClassification362Train = zeros(9,1);
errorClass362Train = zeros(9,1);
errorClassification362Test = zeros(9,1);
errorClass362Test = zeros(9,1);

for trainingPercentage = 10:5:90
    iPercentage = iPercentage + 1;
    %% Split Dataset
    trainSet1 = features(1:round(trainingPercentage/100*648),:);
    testSet2 = features(round(trainingPercentage/100*648)+1:648,:);
    labelsTrain = labels(1:round(trainingPercentage/100*648));
    labelsTest = labels(round(trainingPercentage/100*648)+1:648);

    %% Calculate sizes
    
    sizeTrain = size(trainSet1);
    sizeTest = size(testSet2);
    
    %Class A: label '1', correct movement of cursor
    %Class B: label '0', erroneous movement of cursor
    trainClassA = trainSet1(find(labelsTrain == 1), :); %vector of class A
    trainClassB = trainSet1(find(labelsTrain == 0), :); %vector of class B
    sizeTrainClassA = size(trainClassA);
    sizeTrainClassB = size(trainClassB);
    ratioTrain = sizeTrainClassA(1)/sizeTrainClassB(1);

    testClassA = testSet2(find(labelsTest == 1), :); %vector of class A
    testClassB = testSet2(find(labelsTest == 0), :); %vector of class B
    sizeTestClassA = size(testClassA);
    sizeTestClassB = size(testClassB);
    ratioTest = sizeTestClassA(1)/sizeTestClassB(1);

    %% Classification and optimal TH with TRAINING set

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

        for i = 1:sizeTrain(1)
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

    errorClassification362(it) = 1 - nCorrect362/(sizeTrain(1));
    errorClass362(it) = 0.5*misClassA/sizeTrainClassA(1) + 0.5*misClassB/sizeTrainClassB(1);

    end

    figure('Color','w');
    th = -6:0.09:6;
    plot(th,errorClassification362,'r-');
    hold on;
    plot(th, errorClass362,'b-');
    hold on;
    title(['Error in comparison with chosen threshold (training : ' num2str(trainingPercentage) ' %)'])
    hold on;
    xlabel('Threshold'); ylabel('Error');
    legend('Classification Error','Class Error','Location','northwest');
    box off;
    hold off;

    % Cohérence de prendre la mean juste pour avoir une valeur ???
    optimalTH1 = mean(th(find(errorClassification362 == min(errorClassification362))));
    optimalTH2 = mean(th(find(errorClass362 == min(errorClass362))));

    errorClassification362Train(iPercentage) = min(errorClassification362);
    errorClass362Train(iPercentage) = min(errorClass362);

    % Scatter
    % We keep feature 362
    figure('Color','w');
    title(['Scatter plot with Training Set (training : ' num2str(trainingPercentage) ' %)'])
    hold on;
    scatter(trainClassA(:,362),trainClassA(:,366),'.b');
    hold on;
    scatter(trainClassB(:,362),trainClassB(:,366),'.r');
    xlabel('Feature 362'); ylabel('Feature 366');
    hold on;
    line([optimalTH1 optimalTH1],[-8 8],'Color','k','LineStyle',':');
    hold on;
    line([optimalTH2 optimalTH2],[-8 8],'Color','k','LineStyle','--');
    legend('Samples from Class A (correct)','Samples from Class B (erroneous)','Threshold with minimal classification error','Threshold with minimal class error','Location','best');
    hold off;

    %% Classification of TESTING set with found TH

    sf362 = testSet2(:,362);

    nCorrect362 = 0;
    misClassA = 0;
    misClassB = 0;

    labelf362 = (sf362 > optimalTH2); % element is 1 if Class A

    % Performance of Test
    for i = 1:sizeTest(1)
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

    errorClassification362Test(iPercentage) = 1 - nCorrect362/(sizeTest(1));
    errorClass362Test(iPercentage) = 0.5*misClassA/sizeTestClassA(1) + 0.5*misClassB/sizeTestClassB(1);

    % Scatter
    % We keep feature 362
    figure('Color','w');
    title(['Scatter plot with Testing Set (training : ' num2str(trainingPercentage) ' %)'])
    hold on;
    scatter(testClassA(:,362),testClassA(:,366),'.b');
    hold on;
    scatter(testClassB(:,362),testClassB(:,366),'.r');
    xlabel('Feature 362'); ylabel('Feature 366');
    hold on;
    line([optimalTH1 optimalTH1],[-8 8],'Color','k','LineStyle',':');
    hold on;
    line([optimalTH2 optimalTH2],[-8 8],'Color','k','LineStyle','--');
    legend('Samples from Class A (correct)','Samples from Class B (erroneous)','Threshold with minimal classification error','Threshold with minimal class error','Location','best');
    hold off;
end

trainingPercentage = 10:5:90;
figure('Color','w');

plot(trainingPercentage, errorClassification362Train,'r--');
hold on;
plot(trainingPercentage, errorClass362Train,'b--');
hold on;
plot(trainingPercentage, errorClassification362Test,'r-');
hold on;
plot(trainingPercentage, errorClass362Test,'b-');
xlabel('Training Percentage of Dataset [%]'); ylabel('Error');
legend('Training Classification Error','Training Class Error','Testing Classification Error','Testing Class Error','Location','northwest');
box off;
hold off;