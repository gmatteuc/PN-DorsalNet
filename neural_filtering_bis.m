function [ prate, pcount, pratetime ] = neural_filtering_bis( filter,stimul,alpha,beta,sr,countingwindowlim )

% get filter and stimulus
S=stimul;
F=filter;

% % % % aaa=reshape(filter,[110,110,4]);
% % % % bbb=reshape(stimul(:,45),[110,110,4]);
% % % % figure;
% % % % for i=1:4
% % % %     subplot(1,2,1)
% % % %     imagesc(aaa(:,:,i))
% % % %     subplot(1,2,2)
% % % %     imagesc(bbb(:,:,i))
% % % %     pause(0.3)
% % % % end

% convolve to compute filter response ad apply nonlinearlity
fout=alpha.*((max(S'*F,zeros(size(S,2),1))).^beta)';
% S = unrolled spatial x temporal ... F = unrolled spatial x 1

% pad filter response at the beginning
prate=[fout(1)*ones(10,1);fout'];

% get filter response time vector
pratetime=(1/sr).*(1:(size(S,2)+10));

% integrate over counting window
countingwindowsamples=and(pratetime>countingwindowlim(1),pratetime<countingwindowlim(2));
pcount=nansum(prate(countingwindowsamples));

end

