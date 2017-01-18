function k = find_mnm_param(epsilon);
% Find the Huber's minimax param k
% for a given epsilon

if and(epsilon<=0.2,epsilon>0.1)
	x0 = 1;
elseif and(epsilon<=0.1,epsilon>0.05)
	x0 = 1.7;
elseif	and(epsilon<=0.05,epsilon>0.02)
	x0 = 2.1;
elseif and(epsilon<=0.02,epsilon>0.01)
	x0 = 2.45;
elseif and(epsilon<=0.01,epsilon>0.005)
	x0 = 2.7;
elseif and(epsilon<=0.005,epsilon>=0.001)
	x0 = 3.0;
else
	error(['The value of epsilon =',num2str(epsilon),' is not supported'])	
end;	

k = fzero('huber_connect',x0,[],epsilon);