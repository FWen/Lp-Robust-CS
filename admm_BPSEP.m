function [x,out] = admm_BPSEP(A,y,lamda,xtrue,rho,x0,max_iter);
% admm_BPSEP solves
%
%   minimize || Ax - y ||_2^2 + \lamda || x ||_1
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
if nargin<7
    max_iter = 1000;
end

ABSTOL = 1e-7;

if(isobject(A))
    m = A.m;
    n = A.n;
else
    [m,n]=size(A);
end

if nargin<5
	rho = 1;
end
    
%Initialize
if nargin<6
	x  = zeros(n,1);
else
	x = x0;
end;

v = zeros(m,1);
w = zeros(m,1); 

out.e  = [];
out.et = [];out.f = [];
tic;

xm1 = x;
rhoT = rho;
rho = 1e-5;
for i = 1 : max_iter
    vm1 = v;
    
    if rho<rhoT
        rho =rho*1.03;
    end
           
    %v-step
	tv = A*x-y-w/rho;
    v  = shrinkage_Lp(tv, 2, 1/lamda, rho); 
   
    %x-step
    tao = 0.5; % for orthonornal A
    z = x - tao*(A'*(A*x-y-v-w/rho)); 
    x = sign(z) .* max(abs(z)-tao/rho,0);
    
	%w-step
    Ax = A*x;
	w = w - rho*(Ax - y - v);
  
    
%   out.et = [out.et toc];
%   out.e  = [out.e norm(x-xm1)];
%   out.e  = [out.e norm(x(1:(n-m))-xtrue)/norm(xtrue)];
    
    xm1 = x;
    %terminate when both primal and dual residuals are small
    if (norm(rho*(v-vm1))< sqrt(n)*ABSTOL && norm(Ax-y-v)< sqrt(n)*ABSTOL) 
        break;
    end
end

end
