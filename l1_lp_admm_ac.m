function [x,out] = l1_lp_admm_ac(A,y,lamda,p,xtrue,rho,x0,max_iter);
% l1_lp_admm_ac solves (with smoothing and acceleration)
%
%   minimize || Ax - y ||_p^p + \lamda || x ||_{1, \epsilong}
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
if nargin<8
    max_iter = 2000;
end

ABSTOL = 1e-7;

[m,n]=size(A);
A2 = A'*A;

if nargin<6
	rho = 1e3;
end
    
%Initialize
if nargin<7
	x = zeros(n,1);
else
	x = x0;
end;

v = zeros(m,1);
w = zeros(m,1); 

out.e  = [];
out.et = [];
out.f  = [];
tic;

u = x;
t = 1;
wp1 = w;
ep = 1e-3; %\epsilong
L2 = 1/ep;

%iA = inv(L2*eye(n) + rho*A2);     % used for non-orthonormal A, i.e., A*A' ~= I
iA = (eye(n)-rho/(rho+L2)*A2)/L2;  % used for orthonormal A, i.e., A*A' = I
for i = 1 : max_iter
    
    xm1 = x;
    vm1 = v;
    wm1 = w;
    
    %v-step
	tv = A*u-y-wp1/rho;
    v  = shrinkage_Lp(tv, p, 1/lamda, rho); 
   
    %x-step  
    z = L2*x + rho*(A'*(y+v+wp1/rho)) - x./sqrt(x.^2+ep*ep); 
    x = iA*z;
    %x = (z-rho/(rho+L2)*A'*(A*z))/L2; % for fast computation,e.g, A is a partial DCT function 
    
	%w-step
    Ax = A*x;
	w = wp1 - rho*(Ax - y - v);
  
    tp1 = (1 + sqrt(1+4*t^2))/2;
 	u   = x + (t-1)/(tp1)*(x-xm1);
	wp1 = w + (t-1)/(tp1)*(w-wm1);
    t   = tp1;

    out.et = [out.et toc];
    out.e  = [out.e norm(x-xtrue)/norm(xtrue)];
    
    %terminate when both primal and dual residuals are small
    if (norm(rho*(v-vm1))< sqrt(n)*ABSTOL && norm(Ax-y-v)< sqrt(n)*ABSTOL) 
        break;
    end
    
end

end

