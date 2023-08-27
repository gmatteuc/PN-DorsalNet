function [hmu,hstd] = plot_shaded_mu_std(ax,xvals,mu,std,color,alpha)

%test if correct size
if size(mu,1)~= 1
    mu = mu';
end
if size(std,1)~= 1
    std = std';
end

% initialize x values
if isempty(xvals)
    x = 1:length(mu);
else
    x = xvals;
end
if size(x,1)~= 1
    x = x';
end

% remove  points with mean NaN
nanidx=find(isnan(mu));
mu(nanidx)=[];

std(nanidx)=[];
x(nanidx)=[];

% replace std NaN with zeros
std(isnan(std)) = 0;

% plot patch
hmu=plot(ax,x,mu,'-','Color',color);
uE = mu+std;
lE = mu-std;
yP = [lE,fliplr(uE)];
xP = [x,fliplr(x)];
hstd = patch(ax,xP,yP,1,'facecolor',color,...
    'edgecolor','none',...
    'facealpha',alpha);

end

