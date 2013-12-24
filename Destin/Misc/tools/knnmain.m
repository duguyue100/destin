%% clear histroy

%clc
%clear
%close all

%% init

% load data
% can try other bigger dataset.

%Test=[1,1];

%Train=[2,2;3,3;4,4;2,1;1,2;4,3;3,4];
%Label=[1;2;2;1;1;2;2];
%K=5;
%M=2;

%OutPutBeliefs=load('OutPutBeliefs.txt');

training=load('trainOutput.txt');
testing=load('testOutput.txt');

training=training*10;
testing=testing*10;

label=load('trainingclassLabel.txt');
testLabel=load('testingclassLabel.txt');

%createLabel;

%% processing

indexResult=zeros(length(testLabel),1);
for i=1:length(testLabel)
    knnprob=knn(testing(i,:), training, label, 3, 33);

    %disp('the classification result of test case:');

    %disp('probability distribution:');
    %disp(knnprob);

    %disp('Class belonging:');
    [dummy, index]=max(knnprob);
    indexResult(i)=index;
    %disp(index);
end

avg_accuracy=sum(indexResult==testLabel)/length(indexResult);
disp('Average accuracy: ');
disp(avg_accuracy);