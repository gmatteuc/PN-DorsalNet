function [ output_image ] = apply_mask( input_image, bool_plot )
% [ output_image ] = apply_mask( input_image )
% this function appliea a gaussian masking centred on the image centre

%get binarized version
binp=input_image;
binp=double(not((binp<4) & (binp>-4)));
% find center and size of RF
properties={ 'Centroid', 'MajorAxisLength' };
Istats = regionprops(binp,properties);
if isempty(Istats)
    cm=[size(input_image,1)/2,size(input_image,1)/2];
    ma=max(size(input_image))/6;
else
    cm=Istats.Centroid;
    %ma=4*Istats.MajorAxisLength;
    sigma=min(size(input_image))/6;
    ma=1.5*(sigma); %STD (removed 2.35 factor)
end

% apply gaussian mask
mask = GaussianMask( 1,cm(1),cm(2),ma,ma,0,3); % mask(mask<0.7) = 0;
output_image=input_image.*mask;

% plot
if bool_plot == 1
    figure; imagesc(input_image); colormap(gray); colorbar; caxis([-5,5])
    figure; imagesc(output_image); colormap(gray); colorbar; caxis([-5,5])
    figure; imagesc(binp); colormap(gray); colorbar; caxis([0,1])
    hold on
    scatter(cm(1),cm(2),70,[0,1,0],'fill');
    hold off
end

end

