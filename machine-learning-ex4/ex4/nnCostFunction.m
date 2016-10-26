function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly 
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Input Layer
a1 = [ones(m, 1) X];
% Hidden Layer
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];
% Output Layer
z3 = a2 * Theta2';
a3 = sigmoid(z3);


% Recode labels as vectors containing only 0s and 1s
y_vec = ones(m, num_labels);
labels = 1:num_labels;
for i = 1:m
    y_vec(i,:) = (labels == y(i));
endfor
y = y_vec;

Theta1NoBias = Theta1(:, 2:end);
Theta2NoBias = Theta2(:, 2:end);

% Compute cost over all samples and all labels
J = 0;
for i = 1:m
    for k = 1:num_labels
        J += (-y(i,k) * log(a3(i,k)) - (1 - y(i,k)) * log(1 - a3(i,k)));
    endfor
endfor
reg = lambda / (2 * m) * (sum(sum(Theta1NoBias .^ 2)) + sum(sum(Theta2NoBias .^ 2)));
J = 1/m * J + reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients Theta1_grad and Theta2_grad.
acc_1 = 0;
acc_2 = 0;
for t = 1:m
    % Feedforward
    % Input Layer
    a1 = [1; X(t, :)'];
    % Hidden Layer
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    % Output Layer
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % Compute error of output layer
    delta_3 = a3 - y(t, :)';
    % Compute error of hidden layer
    delta_2 = (Theta2NoBias' * delta_3) .* sigmoidGradient(z2);
    acc_1 += (delta_2 * a1');
    acc_2 += (delta_3 * a2');
endfor

Theta1_grad = 1 / m * acc_1;
Theta2_grad = 1 / m * acc_2;


% Part 3: Implement regularization with the cost function and gradients.
Theta1_grad(:, 2:end) += (lambda / m .* Theta1NoBias);
Theta2_grad(:, 2:end) += (lambda / m .* Theta2NoBias);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
