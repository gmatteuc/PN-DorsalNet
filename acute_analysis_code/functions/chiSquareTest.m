function [chi2stat,p] = chiSquareTest(X)
%% basic chi square test
XSize = size(X);
df = (XSize(1) - 1) * (XSize(2) - 1);
X2 = sum(X,2);
X1 = sum(X,1);
Xtotal = sum(X(:));
E = [];
for currR = 1:length(X(:,1))
    for currC = 1:length(X(1,:))
        E(currR,currC) = (X2(currR)*X1(currC))/Xtotal;
        
    end % col
end %row
chi2stat = sum(sum((X - E).^2 ./ E));
p = 1 - chi2cdf(chi2stat,df);
end
