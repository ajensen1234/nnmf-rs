function [W, H, err] = nnmf_Jessica(V, r, flagStd, flagMethod)
%
% [W, H, err] = nnmf_Jessica(V, r, flagStd, flagMethod)
%
% NNMF: Given a nonnegative matrix V, NNMF finds nonnegative matrix W 
%       and nonnegative coefficient matrix H such that V~WH. The algorithm 
%       solves the problem of minimizing (V-WH)^2 by varying W and H.
%
%       
%       Multiplicative update rules developed by Lee and Seung were used to solve 
%       optimization problem. (see reference below) 
%          D. D. Lee and H. S. Seung. Algorithms for non-negative matrix
%          factorization. Adv. Neural Info. Proc. Syst. 13, 556-562 (2001)
%
%
% INPUTS:       V - nonnegative matrix to be factorized (dimensions nxm)
%               r - integer, number of basis vectors to be used for 
%                   factorization, usually r is chosen to be smaller than n
%                   or m so that W and H are smaller than original matrix V
%               flagStd - flag for unit variance normalization
%                         ( 1 = normalize to input condition std, default )
%                         ( 2 = normalize to some global value of std)
%               flagMethod - flag for choosing NNMF decomposition method
%                            ( 1 = matlab nnmf Lee and Seung algorithm, default )
%                            ( 2 = matlab nnmf least squares algorithm )
%                            ( 3 = lab generated Lee and Seung code )
%                            
%
%
% OUTPUTS:      W - nonnegative matrix containing basis vectors (nxr)
%               H - nonnegative matrix containing coefficients (rxm)
%               err - least square error (V-WH)^2 after optimization
%                     convergence
%
% Large chunk of this code is from code originally created by JLM and GTO:
%   Created: March 15th, 2004 by JLM and GTO 
%   Last modified by GTO on March 2006
%   Last modification: Remove for iteration channels with only zeros for all
%   conditions
%
% Modifications: 3/13/13, JLA: added the ability to use matlabs built-in
%                               nnmf function, some flags for nnmf
%                               algorithm to use, and whether to use
%                               condition std or some global std for unit
%                               variance normalization
%               12/8/16, JLA: when looking for zeros to indicate bad data,
%                             using nansum since bad data is indicated in
%                             the EMG matrix by NaNs. Then, when
%                             re-introducting bad data after extraction,
%                             now put back in NaNs instead of zeros. 
%                             


% set some defaults
if nargin < 3
    flagStd = 1;
    flagMethod = 1;
elseif nargin < 4
    flagMethod = 1;
end



V = V.*(V>0); % Any potential negative entry in data matrix will be set to zero

test=nansum(V,2); % Any potential muscle channnel with only 0's is not included in the iteration 
index=find(test~=0);
ind=find(test==0);
Vnew_m=V(index,:);

test_cond=nansum(V,1); % Any potential condition with only 0's is not included in the iteration 
index_cond=find(test_cond~=0);
ind_cond=find(test_cond==0);
Vnew=Vnew_m(:,index_cond);

%%%%% Scale the input data to have unit variance %%%%%%%%%
if flagStd ==1;  
    stdev = std(Vnew'); %scale the data to have unit variance of this data set   
elseif flagStd ==2;
    global stdev % use this if you want to use the stdev (unit variance scaling) from a different data set
end

Vnew = diag(1./stdev)*Vnew;

if flagMethod == 1

    opts = statset('MaxIter',1000,'TolFun',1e-6,'TolX',1e-4);
    [W,H,err] = nnmf(Vnew,r,'alg','mult','rep',50,'options',opts);
%     [W,H,err] = nnmf(Vnew,r,'alg','mult','rep',50);
    
elseif flagMethod == 2

    [W,H,err] = nnmf(Vnew,r,'alg','als','rep',50);
    
elseif flagMethod == 3

    % Initial conditions.
    [n,m]=size(Vnew);
    H=rand(r,m);
    W=rand(n,r);
    err=sum(sum((Vnew-W*H).^2));
    
    MAX_IT=100000;   % Increased this from 10000 to 100000
    % Error goal - the "err" quantity, as defined, is the squared error.  If we
    % want a 1% mse, then, we want .01*prod(size(V))=.01*n*m.
    ERR_GOAL=.0001*(n*m);
    
    % Update...  For normed data, the max err is n x m
    err_save=[];
    
    
    while err>ERR_GOAL
        
        H_fac=W'*Vnew;
        
        H_fac=H_fac./(W'*W*H);
        H=H.*H_fac;
        
        W_fac=Vnew*H';
        W_fac=W_fac./(W*H*H');
        
        W=W.*W_fac;
        
        err=sum(sum((Vnew-W*H).^2));
        err_save=[err_save;err];
        if length(err_save)>550
            err_change = (err_save(end-500) - err_save(end))/err_save(end-500);
            if abs(err_change)<.01
                disp('NMF.m: Error has changed less than 1% over the last 500 iterations. Exiting.')
                break;
            end
        end
        if length(err_save)>MAX_IT
            break;
        end
    end
    
end

% Re-scale the original data and the synergies; add in zero rows; calculate 
% final error.

%undo the unit variance scaling so synergies are back out of unit variance
%space and in the same scaling as the input data was
Vnew = diag(stdev)*Vnew;
W = diag(stdev)*W;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Synergy vectors normailzation  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


m=max(W);% vector with max activation values 
for i=1:r
    H(i,:)=H(i,:)*m(i);
    W(:,i)=W(:,i)/m(i);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set to NaN the columns or rows that were not included in the iteration
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n_o,m_o]=size(V);

Hnew=[];
Wnew=[];
for l=1:length(ind_cond)
    if ind_cond(l)==1
        Hnew=[zeros(r,1) H];
        H=Hnew;
    elseif ind_cond(l)==m_o
        Hnew=[H zeros(r,1)];
        H=Hnew;
    else 
        for k=1:m_o
            if ind_cond(l)==k
                Hnew=[H(:,1:k-1) zeros(r,1) H(:,k:end)];
                H=Hnew; break
            else
                Hnew=H;
            end
        end
    end
end
for l=1:length(ind)
    if ind(l)==1
        Wnew=[zeros(1,r); W];
        W=Wnew;
    elseif ind(l)==n_o
        Wnew=[W; zeros(1,r)];
        W=Wnew;
    else 
        for k=1:n_o
            if ind(l)==k
                Wnew=[W(1:k-1,:); zeros(1,r); W(k:end,:)];
                W=Wnew; break
            else
                Wnew=W;
            end
        end
    end
end