%RMS calculator for tcp

filename = dir('Img*.mat')
currFileName = filename(1).name;
matstruct=load(currFileName);
mat = vertcat(matstruct.flameTemp);

RMSmat=zeros(size(mat,1),size(mat,2));
RMSav=zeros(size(mat,1),size(mat,2));
for i=1:length(filename)
    currFileName = filename(i).name;
    file = currFileName(1:end-4);

    matstruct=load(currFileName);
    mat = vertcat(matstruct.flameTemp);
    RMSmat=RMSmat+mat(:,:,2).^2;
    RMSav = RMSav+mat(:,:,2);

end
% RMSmat= sqrt((RMSmat)./i )
RMSmat= sqrt((RMSmat)./i - (RMSav./i).^2);



% figure(1)

figure1 = figure('Color',[1 1 1],'OuterPosition',[10 50 800 450]);
colormap(hot);

axes1 = axes('Parent',figure1);
axis off
hold(axes1,'on');

contourf(RMSmat')
c = colorbar(axes1,'northoutside','Ticks',[600 900 1200 1500 1800 2100 2400 2700 3000 3300]);
title(c, 'T [K]');
hold(axes1,'all');
