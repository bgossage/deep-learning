function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

F = X * theta - y;
J = (F' * F) / (2*m);

% Compute the regularization term...
theta_prime = theta';
theta_prime(1)= 0.0;

R = (theta_prime * theta) * lambda/(2.0*m);

J += R;


%Compute the gradients...
grad = (X' * F) / m .+ lambda/m * theta;

xm = X( :,1 );
grad(1) = (xm' * F) / m;







% =========================================================================

grad = grad(:);

end
