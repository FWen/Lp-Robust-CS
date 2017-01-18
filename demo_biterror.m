clear all; clc;close all;

N = 256;
K = 15;
M = 100;

A = randn(M, N);
A = orth(A')'; 

amp_ratio = 10;
x  = amp_ratio*SparseVector(N,K); % the generated sparse signal
y0 = A*x;

figure(1);
subplot(3,4,1);plot(1:length(x),x);xlim([1 length(x)]);
xlabel('(a) Test signal');set(gcf,'outerposition',get(0,'screensize'));
subplot(3,4,2);plot(1:length(y0),y0);
xlabel('(b) Measurements without noise');


% small Gaussian noise
noise = randn(M,1); 
noise = 0.003*noise/std(noise);
y0 = y0 + noise;


% (impulsive) bit errors like noise
impulsive_ratio = 0.2;
biterr_amp = 10*amp_ratio;
kind = randperm(M);
kind = kind(1:M*impulsive_ratio);
biterr = sign(randn(1,M*impulsive_ratio))*biterr_amp;
y = y0;
y(kind) = biterr;

subplot(3,4,3);plot(1:length(y),y);
xlabel('(c) Corrupted measurements');ylim([-biterr_amp-biterr_amp/amp_ratio biterr_amp+biterr_amp/amp_ratio]);

subplot(3,4,4);plot(1:length(y),y - y0);
xlabel('(d) Measurement noise');ylim([-biterr_amp-biterr_amp/amp_ratio biterr_amp+biterr_amp/amp_ratio]);


lamda_min = 1e-3; lamda_max = 1e1;
lamdas    = logspace(log10(lamda_min),log10(lamda_max),20);


%--Lasso-ADMM------------------------------
t0 = tic;
for k = 1:length(lamdas)
    [x_lasso]   = l1_lp_admm(A, y, lamdas(k), 2, x, 100);
    relerr(1,k) = norm(x_lasso - x)/norm(x);   
    xx(:,k,1)   = x_lasso;
end
disp(sprintf('Lasso-ADMM:  elapsed time is %.3f seconds',toc(t0)));
[mv mi] = min(relerr(1,:));
x_lasso = xx(:,mi,1);  
figure(1);subplot(3,4,5);plot(1:length(x_lasso),x_lasso);
xlabel(['(d) Lasso (L1-L2), RelErr=', num2str(mv,'%10.3f')]);
xlim([1 length(x)]);
figure(2); subplot(2,4,1);plot(1:length(x_lasso),x_lasso-x);
xlabel(['(a) Lasso (L1-L2), RelErr=', num2str(mv,'%10.3f')]);
xlim([1 length(x)]);title('Recovery error');
set(gcf,'outerposition',get(0,'screensize'));


%----Lq-min-------------------------------   
t0 = tic;
[x_Lq] = lq(y, A, 1./(2:11),[0; 0.1; 0.2],2);
relerr_LqMin = norm(x_Lq - x)/norm(x);
disp(sprintf('Lq-min:\t\t elapsed time is %.3f seconds',toc(t0)));
figure(1);subplot(3,4,6);plot(1:length(x_Lq),x_Lq);
xlabel(['(e) Lq-Min, RelErr=', num2str(relerr_LqMin,'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,2);plot(1:length(x_Lq),x_Lq-x);
xlabel(['(b) Lq-Min, RelErr=', num2str(relerr_LqMin,'%10.3f')]);
title('Recovery error');xlim([1 length(x)]);%ylim([-0.5 0.5]);


%----YALL1-------------------------------        
t0 = tic;
for k = 1:length(lamdas)
    [x_YALL1] = YALL1_admm(A, y, lamdas(k), 1, x);
    
    % 'YALL1_admm()' is faster than 'yall1()'
    %opts.tol=1e-8; opts.nu = lamdas(k); 
    %x_YALL1 = yall1(A, y, opts);
    
    relerr(2,k) = norm(x_YALL1 - x)/norm(x);
    xx(:,k,2)   = x_YALL1;
end
disp(sprintf('YALL1:\t\t elapsed time is %.3f seconds',toc(t0)));
[mv mi] = min(relerr(2,:));
x_YALL1 = xx(:,mi,2);   
figure(1); subplot(3,4,7);plot(1:length(x_YALL1),x_YALL1);
xlabel(['(f) YALL1, RelErr=', num2str(mv,'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,3);plot(1:length(x_YALL1),x_YALL1-x);
xlabel(['(c) YALL1, RelErr=', num2str(mv,'%10.3f')]);xlim([1 length(x)]);
title('Recovery error');ylim([-1 1]);


%---Huber-FISTA---------------------------------  
t0 = tic;
epsilon = 0.1; nu_est = 0.05;
for k = 1:length(lamdas)
    [x_HFISTA]  = fista_robust_cs(A, y, lamdas(k), epsilon, nu_est, 0, x_YALL1);
    relerr(3,k) = norm(x_HFISTA - x)/norm(x);
    xx(:,k,3)   = x_HFISTA;
end
disp(sprintf('Huber-FISTA: elapsed time is %.3f seconds',toc(t0)));
[mv mi] = min(relerr(3,:));
x_HFISTA = xx(:,mi,3);  
figure(1);subplot(3,4,8);plot(1:length(x_HFISTA),x_HFISTA);
xlabel(['(g) Huber-FISTA, RelErr=', num2str(mv,'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,4);plot(1:length(x_HFISTA),x_HFISTA-x);
xlabel(['(d) Huber-FISTA, RelErr=', num2str(mv,'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');


%----BP-JP-------------------------------- 
t0=tic;
[x_RSC] = YALL1_admm(A, y, 1, 1, x);
relerr_RSC = norm(x_RSC - x)/norm(x);
disp(sprintf('BP-JP:\t\t elapsed time is %.3f seconds',toc(t0)));
figure(1);subplot(3,4,9);plot(1:length(x_RSC),x_RSC);
xlabel(['(f) BP-JP, RelErr=', num2str(relerr_RSC,'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,5);plot(1:length(x_RSC),x_RSC-x);
xlabel(['(c) BP-JP, RelErr=', num2str(relerr_RSC,'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');


%----BP-SEP------------------------------- 
t0=tic;
A1 = [A eye(M)];
A1 = A1/sqrt(2);
y1  = y/sqrt(2);
epsilons = 1e-7*2.^(1:20);
for k=1:length(epsilons)
   [x_bp] = admm_BPSEP(A1, y1, epsilons(k), x, 10);
   relerr_BPSEP(k) = norm(x_bp(1:N) - x)/norm(x);
   xx(:,k,4) = x_bp(1:N);
end
disp(sprintf('BP-SEP: \t elapsed time is %.3f seconds',toc(t0)));
[mv mi] = min(relerr_BPSEP);
xp = xx(:,mi,4); 
figure(1);subplot(3,4,10);plot(1:length(xp),xp);
xlabel(['(j) BP-SEP, RelErr=', num2str(relerr_BPSEP(mi),'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,6);plot(1:length(xp),xp-x);
xlabel(['(f) BP-SEP, RelErr=', num2str(relerr_BPSEP(mi),'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');
%figure(3);semilogy(epsilons,relerr_BPSEP,'r-*','linewidth',2);set(gca,'xscale','log');


%----Lp-ADM (p=0.5)-------------------------       
t0 = tic;
for k = 1:length(lamdas)
    [x_L1Lp05] = l1_lp_admm_ac(A, y, lamdas(k), 0.5, x, 1e4, x_YALL1);
    relerr(5,k) = norm(x_L1Lp05 - x)/norm(x);
    residual(5,k) = norm(y0-A*x_L1Lp05,0.8)^0.8;
    xx(:,k,5)   = x_L1Lp05;
end
disp(sprintf('Lp-ADM (p=0.5): elapsed time is %.3f seconds',toc(t0)));
mi = find(residual(5,:) <= norm(noise,0.8)^0.8, 1, 'last');
if isempty(mi)
    [mv,mi] = min(residual(5,:) - norm(noise,0.8)^0.8);
end
% [mv, mi] = min(relerr(5,:));
x_L1Lp05 = xx(:,mi,5); 
figure(1); subplot(3,4,11);plot(1:length(x_L1Lp05),x_L1Lp05);
xlabel(['(g) Lp-ADM (p=0.5), RelErr=', num2str(relerr(5,mi),'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,7);plot(1:length(x_L1Lp05),x_L1Lp05-x);
xlabel(['(d) Lp-ADM (p=0.5), RelErr=', num2str(relerr(5,mi),'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');


%----Lp-ADM (p=0.8)-------------------------  
t0 = tic;
for k = 1:length(lamdas)
    [x_L1Lp08] = l1_lp_admm_ac(A, y, lamdas(k), 0.8, x, 1e4, x_YALL1);
    relerr(6,k) = norm(x_L1Lp08 - x)/norm(x);
    residual(6,k) = norm(y0-A*x_L1Lp08,0.8)^0.8;
    xx(:,k,6)   = x_L1Lp08;
end
disp(sprintf('Lp-ADM (p=0.8): elapsed time is %.3f seconds',toc(t0)));
mi = find(residual(6,:) <= norm(noise,0.8)^0.8, 1, 'last');
if isempty(mi)
    [mv,mi] = min(residual(6,:) - norm(noise,0.8)^0.8);
end
% [mv, mi] = min(relerr(6,:));
x_L1Lp08 = xx(:,mi,6);
figure(1); subplot(3,4,12);plot(1:length(x_L1Lp08),x_L1Lp08);
xlabel(['(h) Lp-ADM (p=0.8), RelErr=', num2str(relerr(6,mi),'%10.3f')]);xlim([1 length(x)]);
figure(2); subplot(2,4,8);plot(1:length(x_L1Lp08),x_L1Lp08-x);
xlabel(['(e) Lp-ADM (p=0.8), RelErr=', num2str(relerr(6,mi),'%10.3f')]);
xlim([1 length(x)]);ylim([-1 1]);title('Recovery error');

figure(3);semilogy(lamdas,relerr(1,:),'r-',lamdas,relerr(2,:),'r--',lamdas,relerr(3,:),'r-.+',...
   lamdas,relerr(5,:),'b-+',lamdas,relerr(6,:),'b--+','linewidth',2);set(gca,'xscale','log');
legend('Lasso-ADMM','YALL1','Huber-FISTA','Lp-ADM (p=0.5)','Lp-ADM (p=0.8)','Location','SouthWest');
xlim([lamda_min,lamda_max]);grid;
xlabel('\mu');ylabel('RelErr');
