function pp=plot_shaded_distribution(ax,distr_x,distr_y,alpha,color)

% draw shaded distributions
inputdistroutline=distr_y;
xdistr=distr_x;
yP=[inputdistroutline,0,0];
xP=[xdistr,xdistr(end),xdistr(1)];
pp=patch(ax,xP,yP,1,'facecolor',color,...
    'edgecolor','none',...
    'facealpha',alpha);

end

