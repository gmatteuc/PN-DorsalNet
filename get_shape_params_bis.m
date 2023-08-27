function [ aspect_ratio, eccentricity, orientation, contrast_factor, relative_area, lobe_number, fig_handl  ] = get_shape_params_bis( input_image,input_pars, bool_plot )
% [ aspect_ratio, eccentricity, orientation, contrast_factor, relative_area, fig_handl ] = get_shape_params( input_image, bool_plot )
% returns three scalars (computed over the bounding ellipse of the biggest connected region, i.e. lobe) and a figure handle:
% 1) "aspect_ratio" ratio between the minor and the major axis of the bounding ellipse(value is in pixels).
% 2) "eccentricity" of the bounding ellipse (value is in pixels).
% 3) "orientation" angle between the x-axis and the major axis of the ellipse that has the same second-moments as the region (value is in degrees, ranging from -90 to 90 degrees).
% 4) "fig_handl " handle of the figure

% set area limit to consider lobe
lobecount_relareath=input_pars.lobecount_relareath;
% set binarization thereshold
bin_zth=input_pars.bin_zth;

% get area limit in absolute pixel number
fieldareatot=(size(input_image,1)*size(input_image,2));
lobecount_areath=fieldareatot*lobecount_relareath;

% initialize output variable
aspect_ratio=[];
eccentricity=[];
orientation=[];
fig_handl=[];

% get binarized version needed by regionprops
binp=input_image;
binp=(not((binp<bin_zth) & (binp>-bin_zth)));

% call regionprops
properties={ 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'Orientation', 'Area' };
Istats = regionprops(binp,properties);
if isempty(Istats)
    cm=NaN;
    ma=NaN;
    mi=NaN;
    ec=NaN;
    or=NaN;
    su=NaN;
    ar=NaN;
else
    
    % decide witch connected region(s) to use
    for k = 1:length(Istats)
        esizes(k)=Istats(k).Area;
    end
    % [~,mesize_idx]=max(esizes);
    mesize_idxs=find(esizes>=lobecount_areath);
    cm=nan(numel(mesize_idxs),2);
    if not(isempty(mesize_idxs))
        % estract measurements
        for ii=1:numel(mesize_idxs)
            mesize_idx=mesize_idxs(ii);
            cm(ii,:)=Istats(mesize_idx).Centroid; %#ok<*AGROW>
            or(ii)=Istats(mesize_idx).Orientation; % returns a scalar that specifies the angle between the x-axis and the major axis of the ellipse that has the same second-moments as the region. Value is in degrees, ranging from -90 to 90 degrees.
            ma(ii)=Istats(mesize_idx).MajorAxisLength;
            mi(ii)=Istats(mesize_idx).MinorAxisLength;
            ar(ii)=mi(ii)/ma(ii);
            ec(ii)=Istats(mesize_idx).Eccentricity;
            su(ii)=Istats(mesize_idx).Area;
        end
    else
        cm=NaN;
        ma=NaN;
        mi=NaN;
        ec=NaN;
        or=NaN;
        su=NaN;
        ar=NaN;
    end
    
end

% assign outputs
aspect_ratio=median(ar);
eccentricity=median(ec);
orientation=median(or);
relative_area=max(su)./(size(input_image,1)*size(input_image,2));
lobe_number=numel(su);

I = input_image;
% set local contrast filter size
nh=ones(round(min(size(I))/5+1));
% get local contrast
J = rangefilt(I,nh);
% get 0.9 quantile
contrast_factor=quantile(J(:),0.90);

% plot
if bool_plot == 1
    
    %     % visualize input
    %     figure; imagesc(input_image); colormap('gray'); colorbar;
    %     figure; imagesc(binp); colormap('gray'); colorbar;
    
    % visualize ellipses as a sanity check ------------
    fig_handl(1)=figure;
    imshow(binp);
    set(gca,'dataAspectRatio',[1 1 1]);
    s=Istats;
    hold on
    phi = linspace(0,2*pi,50);
    cosphi = cos(phi);
    sinphi = sin(phi);
    for k = 1:length(s)
        xbar = s(k).Centroid(1);
        ybar = s(k).Centroid(2);
        a = s(k).MajorAxisLength/2;
        b = s(k).MinorAxisLength/2;
        theta = pi*s(k).Orientation/180;
        R = [ cos(theta)   sin(theta)
            -sin(theta)   cos(theta)];
        xy = [a*cosphi; b*sinphi];
        xy = R*xy;
        x = xy(1,:) + xbar;
        y = xy(2,:) + ybar;
        if not(isempty(mesize_idxs))
            if k == mesize_idx
                plot(x,y,'g','LineWidth',4);
            else
                plot(x,y,'b','LineWidth',4);
            end
        end
    end
    hold off
    %  -----------------------------------------------
    
    % visualize ellipses as a sanity check ------------
    fig_handl(2)=figure; set(fig_handl(2),'Position',[10,10,1500,1000]);
    subplot(2,2,1); imagesc(J); colorbar; colormap('gray'); axis equal; title('local contrast map');  caxis([0,6]);
    subplot(2,2,2); hist(J(:)); title(['local contrast distribution (contrast factor = ',num2str(contrast_factor,'%.2f'),')']);
    subplot(2,2,3); imagesc(I); colorbar; colormap('gray'); title(['original image']); axis equal; caxis([-6,6]);
    subplot(2,2,4); imagesc(binp); colorbar; colormap('gray'); title(['binarized image']); axis equal;
    
end

end

