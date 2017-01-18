function [x,out] = YALL1_admm(A,y,lamda,xtrue,x0);
% YALL1_admm solves
%
%   minimize || Ax - y ||_1 + \lamda || x ||_1
%
% Inputs:
%	A: sensing matrix
%	y: CS data
%	lamda: regularization parameter 
%	x0: initialization 
%	xtrue: for debug, for calculation of errors
% Outputs
%	x: the CS recovery
%	out.e: the error with respect to the true
%	out.et: time index


%Convergence setup
max_iter = 2000;
ABSTOL   = 1e-7;

if(isobject(A))
    m = A.m;
    n = A.n;
else
    [m,n]=size(A);
end

rho  = 1e3;
    
%Initialize
if nargin<5
	x  = zeros(n,1);
else
	x = x0;
end;

w = zeros(m,1); 
v = zeros(m,1);

out.e  = [];
out.et = [];
tic;

for i = 1 : max_iter
	    
    xm1 = x;
    
	%v-step
	tv = A*x-y-w/rho;
    v  = sign(tv) .* max(abs(tv)-1/lamda/rho, 0);
    
    %x-step
    tao = 0.99; %for orthonornal A
    z = x - tao*(A'*(A*x - y - v - w/rho)); 
    x = sign(z) .* max(abs(z)-tao/rho,0);
    
	%u-step
    Ax = A*x;
	w = w - rho*(Ax - y - v);

    %out.e  = [out.e norm(x-xtrue)/norm(xtrue)];
    %out.e  = [out.e norm(x-xm1)];
    %out.et = [out.et toc];
    
    %terminate when residuals are small
    if (norm(x-xm1)< sqrt(n)*ABSTOL && norm(Ax-y-v)< sqrt(n)*ABSTOL) 
        break;
    end
end

end
