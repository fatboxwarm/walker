% Binary Classification Neural Network
% Luke Schooley
% 03/08/2023

tic

%read data, normalize it, & create three sets of X & y
A=dlmread('trainingSet.txt'); % read all data into matrix A
%normalize each input & output using meanNorm function
[a, meanA, rangeA]=meanNorm(A(:,1));
[b, meanB, rangeB]=meanNorm(A(:,2));
[c, meanC, rangeC]=meanNorm(A(:,3));
[d, meanD, rangeD]=meanNorm(A(:,4));
[e, meanE, rangeE]=meanNorm(A(:,5));
[f, meanF, rangeF]=meanNorm(A(:,6));
[g, meanG, rangeG]=meanNorm(A(:,7));
%create each training set of normalized inputs & outputs
tr=floor(.9*length(A));
X=[a(1:tr),b(1:tr),c(1:tr),d(1:tr),e(1:tr),f(1:tr)];
y=[g(1:tr)];

%other important variables
m=size(X,1);
input_layer_size=size(X,2); 
hidden_layer_size= 64;
output_layer_size=size(y,2);
lambda=0.0001;
alpha=0.01;
nbweightmatrices = 4;
Theta = cell(1, nbweightmatrices); 
initial_Theta = cell(1, nbweightmatrices);

%random initialization of weights
initial_Theta{1} = randInitializeWeights(input_layer_size, hidden_layer_size);
for i = 2 : nbweightmatrices - 1 
   initial_Theta{i} = randInitializeWeights(hidden_layer_size, hidden_layer_size);
end
initial_Theta{nbweightmatrices} = randInitializeWeights(hidden_layer_size, output_layer_size);
% Unroll parameters
initial_nn_params = [initial_Theta{1}(:)]; 
for i = 2 : nbweightmatrices 
    initial_nn_params = [initial_nn_params ; initial_Theta{i}(:)];
end

%% Create "short hand" for the cost function to be minimized
costFunction = @(p) linCostFunction(p, input_layer_size, hidden_layer_size, ...
    output_layer_size, X, y, lambda, alpha, nbweightmatrices);
% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
options = optimset('MaxIter', 1000000);
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

%test and predict
Xcv=[a(tr+1:end),b(tr+1:end),c(tr+1:end),d(tr+1:end),e(tr+1:end),f(tr+1:end)];
ycv=[f(tr+1:end)];
p=linPredict(nn_params, Xcv, hidden_layer_size, input_layer_size, output_layer_size, nbweightmatrices);

%results
actual=reverseMeanNorm(ycv(:,1),meanG,rangeG);
guess=reverseMeanNorm(p(:,1),meanRR,rangeRR);
percentError=(sum(guess-actual))/sum(actual);
delta=mean(abs(actual-guess));

toc