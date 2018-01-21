function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 

   g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% Matrix of ones to handle the non-scalar case...
   id = ones( size(z) );

% Element-wise to handle both scalar and non-scalar inputs...
   g = id ./ ( id .+ exp( -z ) );


% =============================================================

end
