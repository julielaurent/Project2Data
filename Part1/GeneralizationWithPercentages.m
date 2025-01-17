close all;
clear all;
clc;

load('dataset_ERP.mat');
BestFeature = 681;
OtherFeature = 531;
ratioOrigin = 516/136;
iPercentage = 0;
errorClassificationTrain = zeros(17,1);
errorClassTrain = zeros(17,1);
errorClassificationTest = zeros(17,1);
errorClassTest = zeros(17,1);
ratioDifference = zeros(17,1);

for trainingPercentage = 10:5:90
    iPercentage = iPercentage + 1;
    %% Split Dataset
    randFeatures = randperm(648);
    trainSet1 = features(randFeatures(1:round(trainingPercentage/100*648)),:);
    testSet2 = features(randFeatures(round(trainingPercentage/100*648)+1:end),:);
    labelsTrain = labels(randFeatures(1:round(trainingPercentage/100*648)));
    labelsTest = labels(randFeatures(round(trainingPercentage/100*648)+1:end));

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
    
    % Caluculate ratio difference
    ratioDifference(iPercentage) = abs(ratioOrigin - ratioTrain) + abs((ratioOrigin - ratioTest));
    
    %% Classification and optimal TH with TRAINING set
    errorClassification = zeros(134,1);
    errorClass = zeros(134,1);

    sf = trainSet1(:,BestFeature);
    it = 0;
    
    for tf = -6:0.09:6
    
        nCorrect = 0;
        misClassA = 0;
        misClassB = 0;
        it = it + 1;

        labelf = (sf < tf); % element is 1 if Class A

        for i = 1:sizeTrain(1)
            if (labelsTrain(i) == labelf(i))
                nCorrect = nCorrect + 1;
            else
                if (labelsTrain(i) == 1)
                    misClassA = misClassA + 1;
                else
                    misClassB = misClassB + 1;
                end
            end
        end

    errorClassification(it) = 1 - nCorrect/(sizeTrain(1));
    errorClass(it) = 0.5*misClassA/sizeTrainClassA(1) + 0.5*misClassB/sizeTrainClassB(1);

    end

    figure('Color','w');
    th = -6:0.09:6;
    plot(th,errorClassification,'r-');
    hold on;
    plot(th, errorClass,'b-');
    hold on;
    title(['Error in comparison with chosen threshold (training : ' num2str(trainingPercentage) ' %)'])
    hold on;
    xlabel('Threshold'); ylabel('Error');
    legend('Classification Error','Class Error','Location','northwest');
    box off;
    hold off;

    optimalTH1 = mean(th(find(errorClassification == min(errorClassification))));
    optimalTH2 = mean(th(find(errorClass == min(errorClass))));

    errorClassificationTrain(iPercentage) = min(errorClassification);
    errorClassTrain(iPercentage) = min(errorClass);

    % Scatter
    % We keep best feature
    figure('Color','w');
    title(['Scatter plot with Training Set (training : ' num2str(trainingPercentage) ' %)'])
    hold on;
    scatter(trainClassA(:,BestFeature),trainClassA(:,OtherFeature),'.b');
    hold on;
    scatter(trainClassB(:,BestFeature),trainClassB(:,OtherFeature),'.r');
    xlabel(['Feature ' num2str(BestFeature)]); ylabel(['Feature ' num2str(OtherFeature)]);
    hold on;
    line([optimalTH1 optimalTH1],[-8 8],'Color','k','LineStyle',':');
    hold on;
    line([optimalTH2 optimalTH2],[-8 8],'Color','k','LineStyle','--');
    legend('Samples from Class A (correct)','Samples from Class B (erroneous)','Threshold with minimal classification error','Threshold with minimal class error','Location','best');
    hold off;

    %% Classification of TESTING set with found TH

    sf = testSet2(:,BestFeature);

    nCorrect = 0;
    misClassA = 0;
    misClassB = 0;

    labelf = (sf < optimalTH2); % element is 1 if Class A

    % Performance of Test
    for i = 1:sizeTest(1)
        if (labelsTest(i) == labelf(i))
            nCorrect = nCorrect + 1;

        else
            if (labelsTest(i) == 1)
                misClassA = misClassA + 1;
            else
                misClassB = misClassB + 1;
            end
        end
    end

    errorClassificationTest(iPercentage) = 1 - nCorrect/(sizeTest(1));
    errorClassTest(iPercentage) = 0.5*misClassA/sizeTestClassA(1) + 0.5*misClassB/sizeTestClassB(1);

    % Scatter
    % We keep best feature
    figure('Color','w');
    title(['Scatter plot with Testing Set (training : ' num2str(trainingPercentage) ' %)'])
    hold on;
    scatter(testClassA(:,BestFeature),testClassA(:,OtherFeature),'.b');
    hold on;
    scatter(testClassB(:,BestFeature),testClassB(:,OtherFeature),'.r');
    xlabel(['Feature ' num2str(BestFeature)]); ylabel(['Feature ' num2str(OtherFeature)]);
    hold on;
    line([optimalTH1 optimalTH1],[-8 8],'Color','k','LineStyle',':');
    hold on;
    line([optimalTH2 optimalTH2],[-8 8],'Color','k','LineStyle','--');
    legend('Samples from Class A (correct)','Samples from Class B (erroneous)','Threshold with minimal classification error','Threshold with minimal class error','Location','best');
    hold off;
end

trainingPercentage = 10:5:90;
figure('Color','w');

plot(trainingPercentage, errorClassificationTrain,'r--');
hold on;
plot(trainingPercentage, errorClassTrain,'b--');
hold on;
plot(trainingPercentage, errorClassificationTest,'r-');
hold on;
plot(trainingPercentage, errorClassTest,'b-');
xlabel('Training Percentage of Dataset [%]'); ylabel('Error');
legend('Training Classification Error','Training Class Error','Testing Classification Error','Testing Class Error','Location','northwest');
box off;
hold off;

%% Find ratio closest to original ratio
percentage = 10:5:90;
idx = find(ratioDifference == min(ratioDifference));
optimalTrainingPercentage = percentage(idx);