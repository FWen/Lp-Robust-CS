function [y,y_dash] = minimax_score3(r,epsilon,nu,k);
%[y,y_dash] = minimax_score3(r,nu,epsilon);
% Returns Huber's minimax function
% Similar to the previous version minimax_score2 but adjust nu
% due to re-scaling the objective function to match with purely
% quadratic loss
% Inputs:
%	+ r : points at which the score function is evaluated 
%	+ nu: the scale factor
%	+ epsilon: contamination
% Outputs:
%	+ y: the minimax score function
% Version June 21th  2003
% Version March 16th 2005 USE find_mnm_param
% Nu can be unknown
% Version Sep 2012

if nargin < 3
	nu = scale_estimate(x);
end;

if nargin<4
	kv = find_mnm_param(epsilon);
	k = kv / nu;
end;

y = r.*(abs(r)<=k*nu^2) + k*sign(r)*nu^2.*(abs(r)>k*nu^2);

if nargout > 1
	y_dash = (abs(r)<=k*nu^2);
end;

return;

		
				

  	