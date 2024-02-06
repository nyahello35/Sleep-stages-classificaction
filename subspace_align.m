function[Xs_new , Xt_new] = subspace_align(Xs , Xt , d)
PCAs = pca(Xs);
PCAs = PCAs(:,1:d);


PCAt = pca(Xt);
PCAt = PCAt(:,1:d);


Xa = PCAs * PCAs' * PCAt;
Xs_new = Xs*Xa;
Xt_new = Xt*Xa;