function knnprob=knn(Test, Train, Label, K, M)

% This function describes main KNN algoritm.
% Test is a single feature vector
% Train contains n samples and Label is a n-dimensional row vector
% which represents the corresponding Labels.
% K is an integer which indicates the number of nearest neighbor.
% M is the number of classes.
% return a probability distribution over each classes.

N=size(Train,1);

TestSpace=repmat(Test, N, 1);

Distance=sqrt(sum((TestSpace-Train).^2,2));

[dummy, index]=sort(Distance, 'ascend');

index=index(1:K);

knnprob=zeros(1,M);

for i=1:K
    knnprob(Label(index(i)))=knnprob(Label(index(i)))+1;
end

knnprob=knnprob./K;
    
end