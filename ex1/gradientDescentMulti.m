function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

n = length(theta);
theta1 = theta(1);
theta2 = theta(2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


    h = X * theta;
    
    for i = 1:n
      temp(i) = 0;
    endfor
    
    
    for i = 1:m
      sqrError = (h(i) - y(i));
      for j = 1:n
        temp(j) = temp(j) + (sqrError * X(i,j));
       endfor
    endfor
    
    for x = 1:n
      theta(x) = theta(x) - ((alpha * temp(x))/m);
    endfor
    
    

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
