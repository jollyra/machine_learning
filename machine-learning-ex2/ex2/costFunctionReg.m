function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

hypothesis = sigmoid(X * theta);

% disp('size of hypothesis')
% disp(size(hypothesis))
% disp('size of y')
% disp(size(y))
% disp('size of X')
% disp(size(X))
% disp('theta')
% disp(size(theta))
% disp('lambda')
% disp(size(lambda))

for i = 1:m
    J += -y(i,1) * log(hypothesis(i,:)) - (1 - y(i,1)) * log(1 - hypothesis(i,:));
endfor
J = 1/m * J + (lambda / (2 * m) * sum(theta(2:end, 1).^2));

temp = zeros(size(grad));
% Don't regularize theta(1)
temp(1, 1) = 1/m * sum((hypothesis - y)' * X(:,1));
% Regularize the rest of the thetas
for i = 2:length(grad)
    temp(i, 1) = 1/m * sum((hypothesis - y)' * X(:,i)) + lambda / m * sum(theta(i));
endfor

grad = temp;


% =============================================================

end
