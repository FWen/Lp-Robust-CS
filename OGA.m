function x = OGA(y,A,NbIter)
%OGA Recovery of sparse vectors from incomplete measurements via an orthogonal greedy algorithm

%x=OGA(y,A,NbIter) attempts to find the sparsest solution of the linear system Ax=y 

%y=measurement vector
%A=measurement matrix
%NbIter=number of iterations

%Exploiting a code by Alex Petukhov slightly reshaped by Simon Foucart, July 2008 

[m,N]=size(A);

%normalizing the columns of A
for j=1:N
    d(j)=norm(A(:,j));
end
D=diag(1./d);
An=A*D;

%body of the algorithm
alpha=0.6;
IND=zeros(N,1); x=zeros(N,1);
for i=1:NbIter
xa=abs(An'*y);
mx=max(xa)*0.95;
IND=IND+(xa>mx);
IND=IND>0;
for i=1:50
    dx=(alpha*IND).*(An'*y);
    x=x+dx;
    z=An*dx;
    y=y-z;
end
end
x=D*x;


end