function y = huber_connect(x,epsilon);
% y = huber_connect(x,epsilon);
% Implements huber connecting function
% To find k for a given epsilon
% x = k nu^2 but assume nu=1 hence to find k

y = normpdf(x)./x - erfc(x) - epsilon/2/(1-epsilon);
