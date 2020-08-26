function [x] = adaptive_opt_lambda(A,b,recon_mesh)
%Code written by Calvin as part of the work presented in the manuscript
%"IRL1-minimization improves diffuse optical tomogrpahic imaging" Submitted
%to JBO, 2013 (For any bugs: contact: calvinshaw87@gmail.com)
%Function for choosing the optimal regularization parameter (lambda) for
%Iteratively Reweighted L1 (IRL1)
%Iteratively Reweighted Least-Squares (IRLS)
%Itrative Threshoding Meethod (ITM)
%This function is an implementation of the paper
%J. Feng, C. Qin, K. Jia, D. Han, K. Liu, S. Zhu, X. Yang, J. Tian, 
%``An adaptive regularization parameter choice strategy for multispectral bioluminescence 
%tomography,'' Med. Phys. {\bf 38} (2011).
% Please cite this paper if this code is used in any form



lambda=0.5*max(abs(A'*(b))) %Initial guess for regularization parameter
[nrow ncol]=size(A);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p=0.95; %p-value of the Lp-regularization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sig=1.01;%Sigma as per the manuscript%Change this parameter if the algorithm does not converge

eps=1e-5;%Stopping Threshold
diff=100;
it=1; maxit=100;
while (eps<=diff || it<=maxit)

    x= IRLS_lp(A,b,p,lambda,recon_mesh); %Use this algorithm for IRLS based minimization

% [x] = IRL1_Lp(b, A, lambda, 0.1, 60,p,1e-6);%Use this algorithm for IRL1 based minimization

%   x = ITM(A, A', b, p, 1e-6);%Use this algorithm for ITM based  minimization


f_lambda=norm(A*x-b)^2+lambda*norm(x,p)^p;
f_der=norm(x,p)^p;
D=norm(b)^2;
C=-((D-f_lambda)^2)/(f_der);
T=((D-f_lambda)/(f_der))-lambda;
M=f_lambda-(lambda*f_der);
lambda_prev=lambda;
track_ITM(it)=lambda;
lambda=(C/(sig*M-D))-T

diff=abs((lambda-lambda_prev));
it=it+1;
end
%  plot(track_ITM);
%    save track_ITM
end

