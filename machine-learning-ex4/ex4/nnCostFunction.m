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
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m, 1) X];

labels = 1 : 10;


for i = 1:m
 
   yp = labels == y(i);
   
% Feed forward to get the predicted label...
% Compute the hidden layer values...
   z2 = Theta1 * X(i,:)';
   a2 = [ 1; sigmoid(z2) ];
   
% Compute the output layer (hypothesis)...
   z3 = Theta2 * a2;
   h = sigmoid(z3);

% Add up the cost for each input...
   for k = 1:num_labels
      
      J += -yp(k) * log(h(k)) - (1.0 - yp(k)) * log(1.0 - h(k));
   
   endfor % k
 
endfor % i

J /= m;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

delta_3 = zeros(num_labels,1);
delta_2 = zeros(hidden_layer_size + 1,1);

for i = 1:m
 
   yp = labels == y(i);
   
 % Get the input layer...
   a1 = X(i,:)';
   
% Feed forward to get the predicted label...
% Compute the hidden layer values...
   z2 = Theta1 * a1;
   a2 = [1; sigmoid(z2)]; % Add a_2(0) (bias term)
   
% Compute the output layer (hypothesis)...
   z3 = Theta2 * a2;
   a3 = sigmoid(z3);

% Compute the deltas for the output layer...
   for k = 1:num_labels
      
      delta_3(k) = a3(k) - yp(k);
   
   endfor % k
 
 % Compute the hidden layer deltas...
   grad_z2 = [0;sigmoidGradient( z2 )];  % Add zero-column to be comformant.
   delta_2 = (Theta2' * delta_3) .* grad_z2;
   
% Then remove delta_2(0)...
   delta_2 = delta_2(2:end);
   
% Compute the gradients...
   Theta1_grad = Theta1_grad + delta_2 * a1';
   Theta2_grad = Theta2_grad + delta_3 * a2';
 
endfor % i

Theta1_grad /= m;
Theta2_grad /= m;

% Regularize the wrt the gradient...
Reg = [zeros(hidden_layer_size,1) Theta1(:,2:end)];
Reg *= (lambda/m);
Theta1_grad += Reg;

Reg = [zeros(num_labels,1) Theta2(:,2:end)];
Reg *= (lambda/m);
Theta2_grad += Reg;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Compute the sums of squares of each weight matrix except for the first col...
Reg = sumsq(Theta1(:,2:end)(:)) + sumsq(Theta2(:,2:end)(:));

Reg *= (lambda/(2.0*m));

J += Reg


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
