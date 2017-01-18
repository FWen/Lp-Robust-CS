function [x,f,e,et] = fista_robust_cs(H,y,lambda,epsilon,nu,beta,x0,xtrue);
% FISTA_ROBUST_CS computes robust CS estimates using FISTA method
% Model y = H*x + n
%       x is sparse, n is impulsive
% Optimization problem
%  	min rho(y-Hx) + lambda ||x||_1 + (beta/2) ||x||_2^2
% Syntax: [x,f] = fista_robust_cs(H,y,lambda,epsilon,nu,beta);
% Inputs:
%	H: equivalent to Phi
%	y: CS data
%	lamda: regularization parameter 
%	epsilon,nu: the Huber parameters
%	beta: elastic-net type of regularization
%	x0 (optional): initialization
% 	xtrue (optional): for computing errror
	
% Outputs
%	xhat: the robust CS estimate
%	f   : value of objective function at iterations	
%	e : l2-norm error wrt xtrue
% Version:
%	Sep 2012


%DEBUG OPTION
if nargout>1
	quiet = 0;
else
	quiet = 1;
end;

%Compute Lipschitz constant
L = 2;
if(isobject(H))
    M = H.m;
    N = H.n;
%     L = 2;
else
    [M,N]=size(H);
    At = H';
    A2 = H'*H;
end


%Convergence setup
conv_crit = 1e-5;
max_iter  = 500;
min_iter  = 2;


if nargin<7
	x = zeros(N,1);
else
	x = x0;
end;

if nargin<6
	beta = 0;
end;

%Initialize
t = 1;
u = x;
iter = 0;

if ~quiet tic; end;

kv = find_mnm_param(epsilon);
kappa = kv / nu;

%Wfig = waitbar(0,'FISTA Algorithm is running ');
        

if ~quiet
	num_et = 11;
	initial_error = norm(x-xtrue);
	et = zeros(num_et,1);
	emark = initial_error*logspace(0,-10,11);
	etIdx = 1;
	tic;
end;

%Iteratively update the estimate from
for iter = 1 : max_iter

	%waitbar(iter/max_iter,Wfig)

	if ~quiet
		f(iter) = sum(minimax_funct3(y-H*x,epsilon,nu,kappa)) + lambda*sum(abs(x));	
		e(iter) = norm(x-xtrue);
		if etIdx<=num_et
			if e(iter)<emark(etIdx)
				et(etIdx) = toc;
				etIdx = etIdx + 1;
			end;
		end;
	end;

	xm1 = x;	
	
	%v = u - (1/L)*(H'*(H*u) - H'*y); %For normal CS

	v =  u - (1/L)*(H'*minimax_score3(H*u-y,epsilon,nu,kappa));	%For robust CS

	x = sign(v).*max(abs(v)-lambda/(L+beta),0);

	tp1 = (1 + sqrt(1+4*t^2))/2;

	u = x + (t-1)/(tp1)*(x-xm1);

	t = tp1;	
	

	%Check for convergence
	if norm(x-xm1)<conv_crit*norm(x)		
		if iter>=min_iter
			if ~quiet
				fprintf('fista_robust_cs terminates early at interation = %d \n',iter);
			end;
            %iter
			break;
		end;
	end;	
	        
	
end;

%close(Wfig);

if ~quiet
	run_time = toc;

	if iter==max_iter
		disp('fista_robust_cs reached maximum iterations');		
	end;
	figure(10);
	plot(f);xlabel('iterations');ylabel('cost function');
	title('Convergence behaviour of fista-robust-cs');
	fprintf('Computational time = %d seconds \n',run_time);
end;


return


