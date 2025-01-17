close all;
clear all;
clc;

load('dataset_ERP.mat');
BestFeature = 681;
OtherFeature = 531;
%ratioOrigin = 516/136;
iPercentage = 0;
errorClassTrain = zeros(17,1);
errorClassTest = zeros(17,1);

for trainingPercentage = 10:5:90
    iPercentage = iPercentage + 1;
    %% Randomize and Split Dataset
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

    testClassA = testSet2(find(labelsTest == 1), :); %vector of class A
    testClassB = testSet2(find(labelsTest == 0), :); %vector of class B
    sizeTestClassA = size(testClassA);
    sizeTestClassB = size(testClassB);
    
    ratioTrain(iPercentage) = sizeTrainClassA(1)/sizeTrainClassB(1);
    ratioTest(iPercentage) = sizeTestClassA(1)/sizeTestClassB(1);
    
    labelsOutputTrain = zeros(sizeTrain(1),1);
    labelsOutputTest = zeros(sizeTest(1),1);
    
    %% Classification and optimal TH with TRAINING set
  
    
    sfTrain = trainSet1(:,BestFeature);
    errorClass = zeros(121,1);
    it = 0;
    labelsOutputTrain = 0;
    
    for tf = -6:0.1:6
    
        it = it + 1;

        labelsOutputTrain = (sfTrain < tf); % element is 1 if Class A
        errorClass(it) = classerrorOriginal(labelsTrain, labelsOutputTrain);
     
    end
    
    tf = -6:0.1:6;
    optimalTH = mean(tf(find(errorClass == min(errorClass))));
    errorClassTrain(iPercentage) = min(errorClass);
    
%     % Scatter
%     figure('Color','w');
%     title(['Scatter plot with Training Set (training : ' num2str(trainingPercentage) ' %)'])
%     hold on;
%     scatter(trainClassA(:,BestFeature),trainClassA(:,OtherFeature),'.b');
%     hold on;
%     scatter(trainClassB(:,BestFeature),trainClassB(:,OtherFeature),'.r');
%     xlabel(['Feature ' num2str(BestFeature)]); ylabel(['Feature ' num2str(OtherFeature)]);
%     hold on;
%     line([optimalTH optimalTH],[-8 8],'Color','k','LineStyle','--');
%     legend('Samples from Class A (correct)','Samples from Class B (erroneous)','Threshold with minimal class error','Location','best');
%     hold off;

    %% Classification of TESTING set with found TH
    sfTest = testSet2(:,BestFeature);
    labelsOutputTest = (sfTest < optimalTH); % element is 1 if Class A
    errorClassTest(iPercentage) = classerrorOriginal(labelsTest, labelsOutputTest);
%     % Scatter
%     figure('Color','w');
%     title(['Scatter plot with Testing Set (training : ' num2str(trainingPercentage) ' %)'])
%     hold on;
%     scatter(testClassA(:,BestFeature),testClassA(:,OtherFeature),'.b');
%     hold on;
%     scatter(testClassB(:,BestFeature),testClassB(:,OtherFeature),'.r');
%     xlabel(['Feature ' num2str(BestFeature)]); ylabel(['Feature ' num2str(OtherFeature)]);
%     hold on;
%     line([optimalTH optimalTH],[-8 8],'Color','k','LineStyle','--');
%     legend('Samples from Class A (correct)','Samples from Class B (erroneous)','Threshold with minimal class error','Location','best');
%     hold off;
end

trainingPercentage = 10:5:90;
figure('Color','w');

plot(trainingPercentage, errorClassTrain,'b--');
hold on;
plot(trainingPercentage, errorClassTest,'b-');
xlabel('Training Percentage of Dataset [%]'); ylabel('Error');
legend('Training Class Error','Testing Class Error','Location','northwest');
box off;
hold off;