clear all; clc;
rng('default');

N = 512;
M = 200;
K = 30;

gam   = 5e-3; % scale parameter
alpha = 1;   % characteristic exponent

ps   = 0:0.2:2;
N_MC = 50; % times of independent run      
lamdas = logspace(-3,1,20);

for time=1:N_MC     % independent run
    A = randn(M,N);
    A = orth(A')'; 
    x  = 10*SparseVector(N,K); % generate sparse signal

    %---SaS impulsive noise--------------------------------               
    noise = stblrnd(alpha,0,gam,0,M,1);
    y = A*x +  noise;
        
    for np=1:length(ps) % for different p, 0<=p<=2
        [time ps(np)]

        if ps(np)>=1
            for k = 1:length(lamdas); 
                [x_rec, Out] = l1_lp_admm(A, y, lamdas(k), ps(np), x, 1e2);
                relerr(k)  = norm(x_rec - x)/norm(x);
            end
            [mv,mi] = min(relerr);
            RelErr(time,np) = relerr(mi)

        else  % 0=<p<1
            for k = 1:length(lamdas); 
                [x_rec0, Out] = l1_lp_admm(A, y, lamdas(k), 1, x, 1e2);
                relerr0(k)   = norm(x_rec0 - x)/norm(x);
                xx(:,k)      = x_rec0;
            end
            [mv,mi] = min(relerr0);
            x_0 = xx(:,mi);

            for k = 1:length(lamdas); 
                [x_rec] = l1_lp_admm_ac(A, y, lamdas(k), ps(np), x, 2e4, x_0);
                relerr(k)   = norm(x_rec - x)/norm(x);
            end
            [mv,mi] = min(relerr);
            RelErr(time,np) = relerr(mi)

        end

    end

end

AverRelErr = mean(RelErr)

figure(1);
semilogy(ps,AverRelErr,'r-'); grid;
xlabel('p');ylabel('Averaged relative error of recovery'); 
