function [ OSI,DSI,DI,pref_DIR,pref_idx  ] = compute_SIs( tuning_curve )

% find preferred direction
[ pref_DIR, pref_idx ] = randmax( tuning_curve );

% handle indexes wrap-around (normal)
null_idx=pref_idx+6;
if null_idx>12
    null_idx=pref_idx-6;
else
end
orth_idx1=pref_idx+3;
if orth_idx1>12
    orth_idx1=orth_idx1-12;
else
end
orth_idx2=pref_idx-3;
if orth_idx2<=0
    orth_idx2=12-abs(orth_idx2);
else
end

% calculate DSI (normal)
DSI = (tuning_curve(pref_idx)-tuning_curve(null_idx))/(tuning_curve(pref_idx)+tuning_curve(null_idx));
% calculate OSI (normal)
OSI = ((tuning_curve(pref_idx)+tuning_curve(null_idx))-(tuning_curve(orth_idx1)+tuning_curve(orth_idx2)))/((tuning_curve(pref_idx)+tuning_curve(null_idx)));
% calculate DI (normal)
DI = 1-(tuning_curve(null_idx))/(tuning_curve(pref_idx));


end

