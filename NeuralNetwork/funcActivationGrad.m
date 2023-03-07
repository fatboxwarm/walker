function g = funcActivationGrad(z)
%funcActivationGrad derivative of the activation function 
% for a neural net
g  = sigmoid(z).*(1-sigmoid(z));
end