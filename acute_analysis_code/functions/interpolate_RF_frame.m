function [ outputfr ] = interpolate_RF_frame( inputfr, interp_factor )

        % spline interpolation
        [Xobis,Yobis] = meshgrid(1:1:size(inputfr,2), 1:1:size(inputfr,1));
        [Xpbis,Ypbis] = meshgrid(linspace(1,size(inputfr,2),(size(inputfr,2)*interp_factor )),linspace(1,size(inputfr,1),(size(inputfr,1)*interp_factor )));
        outputfr=interp2(Xobis,Yobis,inputfr,Xpbis,Ypbis,'spline');

end

