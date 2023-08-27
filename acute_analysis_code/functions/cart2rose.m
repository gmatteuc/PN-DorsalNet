function [handle]=cart2rose(x,y);
%
% [handle]=cart2rose(x,y);
% plot cartesian graph (x,y) as a rose plot, returns the handle of the plot
%-----------------------------------------------------------------------------------------

data=y;
centers=x;
clear x y

datarose=zeros(size(data,2)*4,1);
counter=1;
for i=1:length(data)
    
    datarose(counter)=0;
    counter=counter+1;
    datarose(counter)=data(i);
     counter=counter+1;
    datarose(counter)=data(i);
    counter=counter+1;
    datarose(counter)=0;
    counter=counter+1;
    
end

centersrose=zeros(size(centers,2)*4,1);
counter=1;
binhwidth=2*pi/(2*length(data));
for i=1:length(centers)
    
    centersrose(counter)=centers(i)-binhwidth;
    counter=counter+1;
    centersrose(counter)=centers(i)-binhwidth;
     counter=counter+1;
    centersrose(counter)=centers(i)+binhwidth;
    counter=counter+1;
    centersrose(counter)=centers(i)+binhwidth;
    counter=counter+1;
    
end

handle=polar(centersrose,datarose);

end