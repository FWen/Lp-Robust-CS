function x = shrinkage_Lp(b, p, lam, L)

max_iter = 20;
ABSTOL1  = 1e-7;
n = length(b);

if (p==0)
    x = b;
    i1 = find(abs(b)<=sqrt(2*lam/L));
    x(i1) = 0;
elseif (p<1 && p>0)
    x    = zeros(n,1);
    ab   = abs(b);
    beta = ( 2*lam*(1-p)/L )^(1/(2-p));
    tao  = beta + lam*p*beta^(p-1)/L;
    i0   = find(ab>tao);
    in   = length(i0);
    b_u  = zeros(in,1);
    x_u  = zeros(in,1);
    if in>0 
        b_u = ab(i0);
        x_u = b_u;
        for k=1:max_iter              
            deta_x = (lam*p*x_u.^(p-1) + L*x_u - L*b_u) ./ (lam*p*(p-1)*x_u.^(p-2) + L);
            x_u    = x_u - deta_x;
            if (k>2 && norm(deta_x) < sqrt(length(x_u))*1e-7 )
                break;
            end
        end
        x_u = x_u .* sign(b(i0));
        x(i0) = x_u;
    end 
elseif (p==1)
    x = sign(b) .* max(abs(b)-lam/L, 0);
elseif (p>1 && p<2)
    x     = L*abs(b)/(p*lam+L);
    i0    = find(x<1);
    
    b1 = b(i0);
    x1 = x(i0).^(1/(p-1));
    Lowerbound = 1e-15;
    h1    = p*lam*Lowerbound^(p-1)+L*Lowerbound-L*abs(b1(find(x1<Lowerbound)));
    x1(find(h1<0)) = Lowerbound;

    x(i0) = x1;
    
    i1  = find(x>0.9*Lowerbound); 
    if ~isempty(i0) 
        x_u = x(i1);
        b_u = abs(b(i1));
        for i=1:max_iter
            g1 = lam*p*(p-1)*x_u.^(p-2) + L;
            g  = p*lam*x_u.^(p-1) + L*x_u - L*b_u;
            deta_x = g./g1;
            x_u  = x_u - deta_x;
            if (i>2 && norm(deta_x) < sqrt(length(x_u))*ABSTOL1 )
                break;
            end
        end
        x(i1) = x_u;
        x = x .* sign(b);
    end
elseif (p==2)
    x = L*b/(2*lam+L);
end

end
