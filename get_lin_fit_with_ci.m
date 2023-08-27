function [ypred,ypred_ci,xpred,opt_b,opt_b_ci,lmodel] = get_lin_fit_with_ci(xtofit,ytofit,xpredlims)

% intialize linear model
lmodel = @(b,x) b(2)*x+b(1);
% fit linear model
[opt_b,R,~,COV,~,~] = nlinfit(xtofit,ytofit,lmodel,rand(2,1)); %rand(2,1)

% find ci for best fit parameters
opt_b_ci = nlparci(opt_b,R,'covar',COV);
% find ci for prediction
xpred=linspace(min(xpredlims),max(xpredlims),100);
[ypred,ypred_ci] = nlpredci(lmodel,xpred,opt_b,R,'covar',COV); %
% flip if result turns to be transposed in order to be same as x input
if not(sum(size(ypred)==size(xpred)))
    ypred=ypred';
    ypred_ci=ypred_ci';
else
end

end

