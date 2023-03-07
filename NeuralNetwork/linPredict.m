function [p, h] = linPredict(nn_params, X, hidden, input, output, nbwm)

%reshape nn_params back into weighted matricies
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
              
h = cell(1, nbwm);
%function activation
h{1} = funcActivation( [ones(size(X,1), 1) X] * Theta{1}' );

for i = 2 : nbwm
    h{i} =  funcActivation( [ones(size(h{i - 1},1), 1) h{i - 1}] * Theta{i}' );
end

%return
p = h{nbwm};