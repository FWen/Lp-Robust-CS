function y = minimax_funct3(x,epsilon,nu,k);
%Compute the minimax function
switch epsilon
	case 0.2
		kv = 0.685;
	case 0.1
		kv = 1.446;
	case 0.05
		kv = 1.924;
	case 0.02
		kv = 2.327;
	case 0.01
		kv = 2.595;
	case 0.005
		kv = 2.8312;
	case 0.001
		kv = 3.3098;
	case 0
		kv = 10; 		%Approximate when little contamination		
	otherwise
		%error('minimax_score()-> Value of epsilon is not supported !');
end;

if nargin<4
	kv = find_mnm_param(epsilon);
	k = kv / nu;
end;

y = x.^2/(2).*(abs(x)<=k*nu^2) + (-k^2*nu^4/2+k*nu^2*abs(x)).*(abs(x)>k*nu^2) ;
