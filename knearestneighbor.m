clear;clc
y = load('Dataset.txt'); % load dataset
x = y(:,1:8); % take 8 features, exclude label
[n1,p] = size(x);  % n1: number of samples, p: number of features

%% Step 1: Standardize data x to X
X=zscore(x);

%% Step 2: Compute sample covariance matrix
R = cov(X);

%% Also can directly compute the sample correlation matrix
R = corrcoef(x);
disp('Sample correlation matrix:')
disp(R)

%% Step 3: Compute eigenvalues and eigenvectors of R
% R is positive semi-definite, so eigenvalues are non-negative
% R is symmetric; MATLAB sorts eigenvalues in ascending order
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
% Eigenvectors must correspond to eigenvalues one-to-one
% Since eigenvalues were reversed, columns of eigenvectors must also be reversed
% rot90 rotates matrix 90 degrees counterclockwise; transposing after achieves column reversal
V=rot90(V)';
disp(V)


%% Compute the values of selected principal components
m1 =input('Enter the number of principal components to keep:  '); % input number of components to save
F = zeros(n1,m1);  % initialize matrix to store principal components (each column is one component)
for i = 1:m1
    ai = V(:,i)';   % extract the i-th eigenvector and transpose to row vector
    Ai = repmat(ai,n1,1);   % repeat this row vector n times to form an n*p matrix
    F(:, i) = sum(Ai .* X, 2);  % note: after weighting standardized data, sum each row
end
iris = F;
iris(:,m1+1) = y(:,9); % append label as the last column of the reduced dataset


%%
K_Max=50;
acc_avg_history=[];
for k=1:K_Max
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
        accuracy(i)=sum(pre_label==test_set(:,m1+1))/sum(test); % compute accuracy
    end
    acc_average=mean(accuracy); % compute mean accuracy
    acc_avg_history=[acc_avg_history acc_average]; % save accuracy for each K value
end
plot(acc_avg_history);xlabel('K');ylabel('Accuracy');title('Cross-validation (N=5) for K selection')


%% Define classification algorithm
function out=knn(test_set,train_set,K)
    [n ,~]=size(test_set);
    [m,~]=size(train_set);
    for i=1:n
        for j=1:m
            % compute distance
            distance(j)=sqrt(sum((test_set(i,:)-train_set(j,1:end-1)).^2));
        end
        [~,index]=sort(distance,'ascend');
        label=train_set(index,end); % sort by distance
        out(i)=mode(label(1:K)); % take the most frequent label among K nearest neighbors
    end
    out=out';
end
