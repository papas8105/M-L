%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%%I have done a little modification to see how underfit and overfit looks
%%like for various lambdas.

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).

data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);

% Put some labels 
hold on;

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;


%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic 
%  regression to classify the data points. 
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 changed that to an array of
% values
lambda = [0,1,5,10,100];

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost1, grad1] = costFunctionReg(initial_theta, X, y, lambda(1));
[cost2, grad2]  = costFunctionReg(initial_theta,X,y,lambda(2));
[cost3, grad3] = costFunctionReg(initial_theta,X,y,lambda(3));
[cost4, grad4] = costFunctionReg(initial_theta,X,y,lambda(4));
[cost5, grad5] = costFunctionReg(initial_theta,X,y,lambda(5));
fprintf('Cost at initial theta (zeros) l =0: %f\n', cost1);
fprintf('Cost at initial theta (zeros) l =1: %f\n', cost2);
fprintf('Cost at initial theta (zeros) l =5: %f\n', cost3);
fprintf('Cost at initial theta (zeros) l =10: %f\n', cost4);
fprintf('Cost at initial theta (zeros) l =100: %f\n', cost5);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and 
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = [0 , 1 , 5 , 10 ,100];

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta1, J1, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda(1))), initial_theta, options);
[theta2, J2, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda(2))), initial_theta, options);
[theta3, J3, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda(3))), initial_theta, options);
[theta4, J4, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda(4))), initial_theta, options);
[theta5, J5, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda(5))), initial_theta, options);
% Plot Boundary
theta_all = {theta1 , theta2 , theta3 , theta4 , theta5}; %cell
for ii = 1 : 5
    plotDecisionBoundary(theta_all{ii}, X, y);
    hold on;
    title(sprintf('lambda = %g', lambda(ii)))
    % Labels and Legend
    xlabel('Microchip Test 1')
    ylabel('Microchip Test 2')
    legend('y = 1', 'y = 0', 'Decision boundary')
    hold off;
end
% Compute accuracy on our training set
p = predict(theta1, X);
fprintf('Train Accuracy (l = 0): %f\n', mean(double(p == y)) * 100);
p = predict(theta2,X);
fprintf('Train Accuracy (l = 1): %f\n', mean(double(p == y)) * 100);
p = predict(theta3,X);
fprintf('Train Accuracy (l = 5): %f\n', mean(double(p == y)) * 100);
p = predict(theta4,X);
fprintf('Train Accuracy (l = 10): %f\n', mean(double(p == y)) * 100);
p = predict(theta5,X);
fprintf('Train Accuracy (l = 100): %f\n', mean(double(p == y)) * 100);


