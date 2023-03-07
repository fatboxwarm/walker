function [J, grad] = linCostFunction(nn_params, input, ...
    hidden, output, X, y, lambda, alpha, nbwm)


%reshape nn_params back into cell of weighted matricies
Theta = cell(1, nbwm);
Theta{1} = reshape(nn_params(1:hidden * (input + 1)), hidden, (input + 1));
inc = hidden * (input + 1); 
for i = 2 : nbwm - 1
    Theta{i} = reshape(nn_params(inc + 1 : inc + hidden * (hidden + 1)), ...
                  hidden, (hidden + 1));
    inc = inc + hidden * (hidden + 1);
end
Theta{nbwm} = reshape(nn_params(inc + 1 : inc + output * (hidden + 1)), ...
                  output, (hidden + 1));
              
%useful variables
m=size(X,1);
J=0; %cost
h_theta = zeros(output, 1); %guess

%calculate cost
h_theta = linPredict(nn_params, X, hidden, input, output, nbwm); %guess
J =(1/m).*(h_theta - y).^2; % compute cost
J=sum(sum(J)); % sums J into one value


%initialization
nbl = nbwm + 1;
a = cell(1, nbl);
z = cell(1, nbl);
delta = cell(1, nbl);
GradAccum = cell(1, nbwm);
for i = 1 : nbwm
    GradAccum{i} = zeros(size(Theta{i}));
end

%feedforward
a{1} = X; %intialize input
for i = 1 : nbwm
    a{i} = [ones(m,1) a{i}]; %add bias
    z{i+1} = a{i} * Theta{i}'; %find z
    a{i+1} = funcActivation(z{i+1}); %next layer
end

%delta values
delta{nbl} = 2 .* (a{nbl} - y) ... 
     .* funcActivationGrad(z{nbl}); %output delta
delta{nbl-1} = delta{nbl} * Theta{nbl-1} .* ... %backwards propagation
    funcActivationGrad([alpha.*ones(length(z{nbl - 1}),1) , z{nbl - 1}]);
for i = nbl - 2 : -1 : 2 %iterate backwards
    delta{i+1} = delta{i+1}(:,2:end); %eliminate bias of delta{i+1}
    delta{i} = delta{i+1} * Theta{i} .* ... %calculate delta{i}
        funcActivationGrad([alpha.*ones(length(z{i}),1) , z{i}]);
end
delta{2} = delta{2}(:,2:end); %eliminate bias of delta{2}

%gradient calculations
for i=1:nbwm
    GradAccum{i} = GradAccum{i}+delta{i+1}'*a{i}; 
    GradAccum{i}(:,1) = GradAccum{i}(:,1) .* (1/m); %bias
    GradAccum{i}(:,2:end) = GradAccum{i}(:,2:end) ...
         .* (1/m) + Theta{i}(:,2:end).*lambda/m; %inputs
end

%unroll gradients
grad = [GradAccum{1}(:)];
for i = 2 : nbwm
    grad = [grad ; GradAccum{i}(:)] ;
end

end

