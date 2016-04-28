function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
% You need to return the following variables correctly 
g = zeros(size(z));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
%can be done much easier but I will use for - loops
for ii = 1 : size(z , 1)
    for jj = 1 : size(z , 2)
        g(ii,jj) = 1  / ( 1 + exp(-z(ii,jj)));
    end
end
% =============================================================
end
