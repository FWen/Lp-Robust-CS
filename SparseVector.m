function [xs] = SparseVector(N,K)
    xs = zeros(N,1);
    xs(randperm(N,K)) = randn(K,1);
    xs = xs/norm(xs);
end