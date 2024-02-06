function [Xs_new, Xt_new] = myTCA_reduceKer(Xs,Xt,kernel,mu,sigma,m,r)
rng(r)
nXs = size(Xs,1);
nXt = size(Xt,1);
X = [Xs ; Xt];
nTotal = size(X,1);
%% Randomly Choose reduced data
%indexs = randi([1,nXs],20,1); %%warning
indext = randi([1,nXt],20,1);
%ReducedXs = Xs(indexs,:); 
ReducedXt = Xt(indext,:);%%warnong
%ReducedX = [ReducedXs;ReducedXt];
ReducedX = [ReducedXt];
%ReducedX = [ReducedXs];
nReducedX = size(ReducedX,1);
%% Define kernel function
linearKer = @(x1,x2)x1*x2';
norm_linearKer = @(x1,x2)(x1*x2')/(norm(x1)*norm(x2));
laplaKer = @(x1,x2)exp(-norm(x1-x2)/sigma);
rbfKer = @(x1,x2)exp(-(norm(x1-x2))^2/(2*sigma^2));
if strcmpi(kernel,'linear')
    for i = 1 : nTotal
        for j = 1 : nReducedX
            K(i,j) = linearKer(X(i,:),ReducedX(j,:));
        end
    end
elseif strcmpi(kernel,'norm_linear')
    for i = 1 : nTotal
        for j = 1 : nReducedX
            K(i,j) = norm_linearKer(X(i,:),ReducedX(j,:));
        end
    end
elseif strcmpi(kernel,'lapla')
    for i = 1:nTotal
        for j = 1:nReducedX
            K(i,j) = laplaKer(X(i,:),ReducedX(j,:));
        end
    end
elseif  strcmpi(kernel,'rbf')
    for i = 1:nTotal
        for j = 1:nReducedX
            K(i,j) = rbfKer(X(i,:),ReducedX(j,:));
        end
    end
else
    error('invalid');
end

%% Define other matrix
H = eye(nTotal) - ones(nTotal)/nTotal; %centerning matrix
L = zeros(nTotal); %L
for i = 1:nXs
    for j = 1:nXs
        L(i,j) = 1/(nXs*nXs);
    end
end

for i = (nXs+1):nTotal
    for j = 1:nXt
        L(i,j) = -1/(nXs*nXt);
    end
end

for i = 1:nXs
    for j = (nXs+1):nTotal
        L(i,j) = -1/(nXs*nXt);
    end
end

for i = (nXs+1):nTotal
    for j = (nXs+1):nTotal
        L(i,j) = 1/(nXt*nXt);
    end
end
%% Find transformation matrix
[E,D] = eig((K'*L*K+mu*eye(nReducedX))\(K'*H*K));
[D,I] = sort(diag(D),'descend');
W = E (:,I(1:m)); %transformation matrix
Xnew = K*W; %embedding data
Xs_new = Xnew(1:nXs,:);
Xt_new = Xnew(nXs+1:nTotal,:);
%[Xs_new, Xt_new] = myTCA(Xs,Xt,'linear',1,1);