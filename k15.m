clear;clc
y = load('Dataset.txt');
x = y(:,1:8);
[n1,p] = size(x);  % n1: number of samples, p: number of features

%% Step 1: Standardize data x to X
X=zscore(x);   % MATLAB built-in standardization: (x - mean(x)) / std(x)

%% Step 2: Compute sample covariance matrix
R = cov(X);

%% Note: Steps 1-2 can be merged — directly compute the sample correlation matrix
R = corrcoef(x);
disp('Sample correlation matrix:')
disp(R)

%% Step 3: Compute eigenvalues and eigenvectors of R
% Note: R is positive semi-definite, so eigenvalues are non-negative
% R is also symmetric; MATLAB sorts eigenvalues in ascending order
% See eig() documentation for details
[V,D] = eig(R);  % V: eigenvector matrix  D: diagonal matrix of eigenvalues


%% Step 4: Compute contribution rate and cumulative contribution rate
lambda = diag(D);  % diag() extracts main diagonal elements (returns a column vector)
lambda = lambda(end:-1:1);  % lambda is sorted ascending, reverse it
contribution_rate = lambda / sum(lambda);  % compute contribution rate
cum_contribution_rate = cumsum(lambda)/ sum(lambda);   % compute cumulative contribution rate; cumsum computes running sum
disp('Eigenvalues:')
disp(lambda')  % transpose to row vector for display
disp('Contribution rates:')
disp(contribution_rate')
disp('Cumulative contribution rates:')
disp(cum_contribution_rate')
disp('Eigenvector matrix corresponding to eigenvalues:')
% Note: eigenvectors must correspond to eigenvalues one-to-one
% Since eigenvalues were reversed, columns of eigenvectors must also be reversed
% rot90 rotates matrix 90 degrees counterclockwise; transposing after achieves column reversal
V=rot90(V)';
disp(V)


%% Compute the values of selected principal components
m1 =input('Enter the number of principal components to keep:  ');
F = zeros(n1,m1);  % initialize matrix to store principal components (each column is one component)
for i = 1:m1
    ai = V(:,i)';   % extract the i-th eigenvector and transpose to row vector
    Ai = repmat(ai,n1,1);   % repeat this row vector n times to form an n*p matrix
    F(:, i) = sum(Ai .* X, 2);  % note: after weighting standardized data, sum each row
end
iris = F;
iris(:,m1+1) = y(:,9);

%%
K_Max=50;
acc_avg_history=[];
rec_avg_history=[];
spec_avg_history=[];
F1_avg_history=[];
auc_history=[];
k=15;
    % K-fold cross-validation
    N=5; % number of folds
    rawrank=randperm(size(iris,1)); % shuffle order
    rand_iris=iris(rawrank,:); % row n of new matrix is row B(n) of original
    indices = crossvalind('Kfold',size(iris,1), N); % randomly split data into N folds
    for i=1:N
        test = (indices == i);
        train = ~test;
        train_set = rand_iris(train, :);
        test_set = rand_iris(test, :);
        pre_label=knn(test_set(:,1:m1),train_set,k); % KNN prediction
        pre_label=pre_label'
        confMat = confusionmat(test_set(:,m1+1), pre_label);
        accuracy(i)= sum(diag(confMat))/sum(confMat(:));
        recall(i) = confMat(1,1)/(confMat(1,1)+confMat(2,1));
        specificity(i) = confMat(1,1) / sum(confMat(1,:));
        precision(i) = confMat(2,2) / sum(confMat(:,2));
        F1(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i));
        [XRF,YRF,TRF,AUCRF] = perfcurve(test_set(:,m1+1),pre_label,1);
        plot(XRF,YRF);
        hold on;
        xlabel('False positive rate');ylabel('True positive rate');
        AUC_SET(i)= AUCRF;
    end
for i = 1:N
    leg_str{i} = ['line',num2str(i)];
end
legend(leg_str)
set(0,'defaultfigurecolor','w')
auc_history=mean(AUC_SET);
  acc_average=mean(accuracy); % compute mean accuracy
  rec_average=mean(recall);
  spec_average=mean(specificity);
  F1_average=mean(F1);

%% Define classification algorithm
function label1=knn(test_set,train_set,k)
    [n ,column1]=size(test_set);
    [m,column]=size(train_set);
        distance1=[];
        distance2=[];
            for i=1:n
                distance1(i,:)=sum((repmat(test_set(i,:),m,1)-train_set(:,1:(column-1))).^2, 2);
            end
            label=[]; % store labels of K nearest neighbors
            weight1=[]; % store first-layer weights
            weight2=[]; % store second-layer weights
            X=sort(distance1,2); % sort distances in ascending order
            for i=1:n
                [a,b]=sort(distance1(i,:));
                for j=1:k
                    label(i,j)=train_set(b(j),column);
                    set1=train_set;
                    set1([b(j)],:)=[]; % remove this training sample
                    distance2(j,:)=sum((repmat(train_set(b(j),1:(column-1)),m-1,1)-set1(:,1:(column-1))).^2, 2); % compute distance from this training sample to all other training samples
                    weight1(i,j)= X(i,j); % first-layer weight
                end
                X1=sort(distance2,2);
                set2=X1(:,1:k);
                for z=1:k
                    [ma]=max(max(distance2(z,:)));
                    distance3=sqrt(sum((test_set(i,:).^-1-train_set(b(j),1:(column-1))).^2));
                    if ma>=distance3
                        weight2(i,z)=100;
                    else
                        weight2(i,z)=0;
                    end
                end
            end

cl=zeros(1,100);
count=1;
cl(1,1)=train_set(1,column); % collect unique class labels
for i=2:m
    A=train_set(i,column);
    flag=0;
    for j=1:count
        if cl(1,j) ==A
          flag=1;
          break;
        end
    end
    if flag==0
        count=count+1;
        cl(1,count)=A;
    end
end

end_count=zeros(n,count); % count weighted votes for each class
for i=1:n
    for j=1:k
        for l=1:count
           if label(i,j)==cl(1,l)
               end_count(i,l)=end_count(i,l)+weight1(i,j).^-1+weight2(i,z);
           end
        end
    end
end

label1=[]; % store final predicted labels
for i=1:n
        k=1;
        num=end_count(i,1);
        for j=2:count
            if num<end_count(i,j)
                k=j;
                num=end_count(i,j);
            end
        end
        label1(i)=cl(1,k);
end
end

function [weight1] = Gaussian(distance1)
mu = 0;
sigma = 0.3;
weight1 = exp(-(distance1-mu).^2/(2*sigma^2));
end
