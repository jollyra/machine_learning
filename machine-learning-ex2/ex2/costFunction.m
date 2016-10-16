function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
hypothesis = sigmoid(X * theta);
% disp('size of hypothesis')
% disp(size(hypothesis))
% disp('size of y')
% disp(size(y))
% disp('size of X')
% disp(size(X))

for i = 1:m
    J += -y(i,1) * log(hypothesis(i,:)) - (1 - y(i,1)) * log(1 - hypothesis(i,:));
endfor

J = 1/m * J;

temp1 = 1/m * sum((hypothesis - y)' * X(:,1));
temp2 = 1/m * sum((hypothesis - y)' * X(:,2));
temp3 = 1/m * sum((hypothesis - y)' * X(:,3));
grad(1) = temp1;
grad(2) = temp2;
grad(3) = temp3;

% =============================================================

end
