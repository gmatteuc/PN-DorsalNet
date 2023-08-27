function [ output_image, crop_ridx, crop_cidx ] = apply_crop( input_image, crop_pixel_size, bool_plot )

% [ output_image ] = apply_crop( input_image, crop_pixel_size, bool_plot )
% this function crops the RF image around its centre of mass
% -------------------------------------------------------------------------

% get binarized version
binp=input_image;
binp=double(not((binp<4) & (binp>-4)));

% find RF center of mass
properties={ 'Centroid', 'MajorAxisLength' };
Istats = regionprops(binp,properties);
if isempty(Istats)
    cm=[size(input_image,1)/2,size(input_image,1)/2]; % arbitrary crop around the center
else
    cm=Istats.Centroid; % crop around RF center of mass
end
 ma=crop_pixel_size; 
 
% choose indexes to crop handling borders
xmin=round(cm(1)+ma/2);
ymin=round(cm(2)+ma/2);
wi=round(ma);
ridx=(ymin-wi:ymin-1);
cidx=(xmin-wi:xmin-1);
if sum(ridx>size(input_image,1))
    ridx=(size(input_image,1)-(wi):size(input_image,1)-1);
elseif sum(ridx<1)
    ridx=(1:wi);
end
if sum(cidx>size(input_image,2))
    cidx=(size(input_image,2)-(wi):size(input_image,2)-1);
elseif sum(cidx<1)
    cidx=(1:wi);
end

% values to be returned
output_image=input_image(ridx,cidx);
crop_ridx=ridx;
crop_cidx=cidx;

% plot diagnostic image
if bool_plot == 1
    figure; imagesc(input_image); colormap(gray); colorbar; caxis([-5,5])
    figure; imagesc(output_image); colormap(gray); colorbar; caxis([-5,5])
    figure; imagesc(binp); colormap(gray); colorbar; caxis([0,1])
    hold on
    scatter(cm(1),cm(2),70,[0,1,0],'fill');
    scatter(xmin,ymin,70,[0.6,0.6,1],'fill');
    plot([cidx(1),cidx(end)],[ridx(1),ridx(1)],'Color','g','LineWidth',2)
    plot([cidx(1),cidx(1)],[ridx(1),ridx(end)],'Color','g','LineWidth',2)
    plot([cidx(1),cidx(end)],[ridx(end),ridx(end)],'Color','g','LineWidth',2)
    plot([cidx(end),cidx(end)],[ridx(1),ridx(end)],'Color','g','LineWidth',2)
    hold off
end

end

