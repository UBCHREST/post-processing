clear all; close all; clc;

%cd('Test1') %change director here
%Make directories to store temperature and soot files
cd ..
mkdir Temperature
mkdir fv
mkdir HeatFlux
mkdir h5files
cd('Images')

filename = dir('*.tiff')
imax=length(filename);
mmperpixel=9.64/42; %meas sample length per pixel amt
imageperdraw=200; %set up the number of images per mask
dt=2500;
dxpix=0.25
sample_width=7.96;
t0=-25000;%first timeshot
i=200;
while i<imax
    idx=0;
    currFileName = filename(i).name;
    image = imread(currFileName);
    % image = imread('Img759.tiff');
    % I = image(50:end,25:380,:);
    % image=image(end-100:end,:,:);
    figure(999);
    h = imshow(image);
    e = drawpolygon;
    vertexX = size(image,1) - floor(e.Position(:,2)) + 1;
    vertexY = floor(e.Position(:,1)) + 1;
    %     BW = createMask(e, h);
    %     complementBW = imcomplement(BW);
    %     image = image.*uint8(complementBW);
    while idx < imageperdraw && i<imax+1
        currFileName = filename(i).name;
        image = imread(currFileName);
        % image = imread('Img759.tiff');
        % I = image(50:end,25:380,:);
        % image=image(end-100:end,:,:);
        BW = createMask(e, h);
        complementBW = imcomplement(BW);
        image = image.*uint8(complementBW);

        pat = digitsPattern;
        framestring= extract(currFileName,pat);
        t=(str2double( framestring )-t0)/dt;
        dotcp(currFileName,image,vertexX,vertexY,BW,mmperpixel,dxpix,t,sample_width,dt)

        idx=idx+1;
        i=i+1;
    end
end


function []=dotcp(filename,image,vertexX,vertexY,BW,mmperpixel,dxpix,timeshot,sample_width,dt)
%% Reading in image and creating a mask



%%
dt = log(1/(2500*2));   %reference f/22 - Sid, This is the log of the exposure time while also accounting for aperture/ND filter. - Elektra
image_red = image;%(:,:,1,:);

img_interest = reshape(image_red, [size(image_red, 1)*size(image_red,2), size(image_red, 3)]);


%Loading physical calibrated function at 700C
prf_ref = load('PRF_Color.dat');
% prf_ref = prf_ref(:,1);

%%
%Obtaining the E*dt* value for images @ 725C from 725C relative response
% Edt_bulb = rrf_bulb(img_interest+1,1);
for i = 1:size(img_interest, 2)-1
    Edt_bulb(:,i) = prf_ref(img_interest(:,i)+1,i);
end

lnE = Edt_bulb - dt(1);
E = exp(lnE);

%Using Planck's law to obtain temperature from the obtained physical
%irradiance
%Define Constants
t = 1/dt ; %Time between Frames
%Setting the constants required for Planck's law
c = 3e8; %Speed of light
h = 6.626e-34; %Planck's constant
k = 1.3806e-23; %Boltzmann Constant

%Planck's first and second constant
c1 = 2*pi*h*c*c;
c2 = h*c/k;

%Filter central wavelength in meters
lambdaR = 650e-9;
lambdaG = 532e-9;

%Temperature in Kelvin calculation from Planck's law
% T2 = (c2./lambda).*(1./(log((c1./(E.*lambda.^5.*10^9))+1)));
T2 = (c2.*(1/lambdaR - 1./lambdaG))./(log(E(:,2)./E(:,1)) + log((lambdaG/lambdaR)^6) + log(4.24/4.55));
%4.24 and 4.55 are empirical optical constants for refractive index for the red and green channels, respectively.

%Temperature in Celcius
% T = T2 - 273;

Temp_matrix = zeros(size(image_red));
Temp_matrix = T2;
Temp = ((reshape(Temp_matrix, [size(image,1), size(image,2), 1])));

mstuff = 1:size(image,2);
parfor k = 1:size(image,1)
    for m = mstuff
        %         if (ans(i,j,1) == 0)
        %             Temp(i,j) = 273;
        %         end
        if (image(k,m,1) <= 0)
            Temp(k,m) = 298;
        end
    end
end


Temp(Temp > 3500) = max(Temp(Temp < 3500));
Temp = flipud(Temp);
BWflip = flipud(BW);
[row col] = find(BWflip == 1);


figure1 = figure('Color',[1 1 1],'OuterPosition',[10 50 800 450],'Visible','off');
colormap(hot);

% Create axes
axes1 = axes('Parent',figure1);
axis off
hold(axes1,'on');

% Create contour
contourf(Temp, 40);
patch(vertexY, vertexX, 'white');
patch(vertexY, vertexX, 'green', 'FaceAlpha', 0.5);
% contour(zdata1,...
%     'LevelList',[298 425.411402977966 552.822805955932 680.234208933897 807.645611911863 935.057014889829 1062.46841786779 1189.87982084576 1317.29122382373 1444.70262680169 1572.11402977966 1699.52543275762 1826.93683573559 1954.34823871356 2081.75964169152 2209.17104466949 2336.58244764745 2463.99385062542 2591.40525360339 2718.81665658135 2846.22805955932]);

box(axes1,'on');
axis(axes1,'tight');
% Set the remaining axes properties
set(axes1,'CLim',[1800 3300],'DataAspectRatio',[1 1 1],'FontName','Arial',...
    'FontSize',12,'LineWidth',2.5,'XTick',...
    [1 51.875 103.75 155.625 207.5 259.375 311.25 363.125 414],'XTickLabel',...
    {'0','10','20','30','40','50','60','70','80'},'YTick',[]);
% Create colorbar
c = colorbar(axes1,'northoutside','Ticks',[1800 2100 2400 2700 3000 3300]);
title(c, 'T [K]');
hold(axes1,'all');

cd ..
cd('Temperature')
saveas(figure1,append(filename(1:end-4),"jpg"))
cd ..
cd('Images')
%% CONSTRUCTING A 3D BOX FOR FLAME
flameTemp = 298.*ones(size(image,1), size(image,2), 20);
parfor k = 2:19
    flameTemp(:,:,k) = Temp;
end

% Computing the soot volume fraction with path length as the width of the
% sample
pathlength = 7.96e-3;

C0_red = 4.24;

Ilambda = reshape(E(:,1), size(Temp));

sootFrac = zeros(size(Temp));

jstuff = 1:size(Temp,2);
parfor i = 1:size(Temp,1)
    for j = jstuff
        if (Temp(i,j) > 400)
            sootFrac(i,j) = (-lambdaR.*1e9./(C0_red.*pathlength))*log(1 - (Ilambda(i,j)*lambdaR^5*exp(c2/(lambdaR*Temp(i,j))))/c1);
        else
            sootFrac(i,j) = 0;
        end
    end
end
% sootFrac(sootFrac > 300e-6) = mean(sootFrac(sootFrac < 200e-6));

fv = zeros(size(flameTemp));
parfor k = 2:19
    fv(:,:,k) = sootFrac;
end

flameTemp = permute(flameTemp, [2, 1, 3]);
fv        = permute(fv, [2, 1, 3]);

c = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2];

c3 = figure('Color',[1 1 1],'OuterPosition',[10 50 800 450],'Visible', 'off');
colormap(hot);

% Create axes
axes1 = axes('Parent',c3);
hold(axes1,'on');

% Create contour
contourf(log(sootFrac), 20);
patch(vertexY, vertexX, 'green', 'FaceAlpha', 0.5);
box(axes1,'on');
axis(axes1,'tight');
% Set the remaining axes properties
set(axes1,'CLim',[log(10^-5) log(10^-1)],'DataAspectRatio',...
    [1 1 1],'FontSize',12,'LineWidth',2.5,'XTick',...
    [1 51.875 103.75 155.625 207.5 259.375 311.25 363.125 414],'XTickLabel',...
    {'0','10','20','30','40','50','60','70','80'},'YTick',[]);
% Create colorbar
h = colorbar(axes1,'northoutside',...
    'Ticks',[log(10^-5) log(10^-4) log(10^-3) log(10^-2) log(10^-1)],...%log(10^-7) log(10^-6)
    'TickLabels',{'10^1','10^2','10^3','10^4','10^5'}); %'10^{-1}','10^0',
title(h, 'fv [ppm]');

% mapping everything into the chrest coordinate system
xcrop=0:mmperpixel:((size(flameTemp,1)+400)*mmperpixel);
ycrop=0:mmperpixel:((size(flameTemp,2)+200)*mmperpixel);
Tempcrop=zeros(length(xcrop),length(ycrop));
Tempcrop=Tempcrop+298.15;
fvcrop=zeros(length(xcrop),length(ycrop));

% figure(7)
% imshow(flameTemp(:,:,2)./(max(max(flameTemp(:,:,2)))))

for i=1:size(flameTemp,1)
    for n=1:size(flameTemp,2)
        Tempcrop((i),n)=flameTemp(i,n,2);
        fvcrop((i),n)=fv(i,n,2);
    end
end
Tempcropplot=Tempcrop./(max(max(Tempcrop)));
% figure(8)
% imshow(Tempcropplot)


xvect=0:dxpix:100-dxpix;
yvect=0:dxpix:25.5-dxpix;
zvect=0:dxpix:25.5-dxpix;

Tempmat=zeros(length(xvect),length(yvect),length(zvect));
fvmat=zeros(length(xvect),length(yvect),length(zvect));
zcent=12.75;

nstuff=1:length(yvect);
qstuff=1:length(zvect);
parfor i=1:length(xvect)
    for n= nstuff

        Tinterp=interp2(xcrop,ycrop,Tempcrop',xvect(i),yvect(n));
        fvinterp=interp2(xcrop,ycrop,fvcrop',xvect(i),yvect(n));
        for q=qstuff
            if (zvect(q)>(zcent-sample_width/2) && zvect(q)<(zcent+sample_width/2))
                    Tempmat(i,n,q)=Tinterp;
                    fvmat(i,n,q)=fvinterp;
            end
        end
    end
end
Tempmatplot=Tempmat./(max(max(max(Tempmat))));
% figure(9)
% imshow(Tempmatplot(:,:,25))



cd ..
cd('fv')
saveas(c3,append(filename(1:end-4),"jpg"))
save(append(filename(1:end-4),"soot.mat"),'fv');
cd ..
cd('Temperature')
save(append(filename(1:end-4),"mat"),'flameTemp');
cd ..
cd('HeatFlux')
save(append(filename(1:end-4),"mat"),'flameTemp');
save(append(filename(1:end-4),"soot.mat"),'fv');
cd ..
cd('h5files')

xstart=0;
ystart=0;
zstart=-0.01275;
%     assume discretization is uniform, dx=dy=dz, get dy from chamber height (1in)
dy=0.0005;
dx=0.0005;
dz=0.0005;

xend=0.1;
yend=0.0255;
zend=0.01275;

%files to write
startm=[xstart ystart zstart];
endm=[xend yend zend];
dxm=[dx dy dz];

file ="highspeed"+sprintf('%.5f',timeshot)
file=strrep(file,'.','_');
hdfname=append(file,'.h5');

if isfile(hdfname)==false
    h5create(hdfname,'/data/fields/fv',size(fvmat))
    h5create(hdfname,'/data/fields/temperature',size(Tempmat))
    h5create(hdfname,'/data/grid/start',[1 3])
    h5create(hdfname,'/data/grid/end',[1 3])
    h5create(hdfname,'/data/grid/discretization',[1 3])
end

h5write(hdfname,"/data/fields/fv",fvmat)
h5write(hdfname,"/data/fields/temperature",Tempmat)
h5write(hdfname,"/data/grid/start",startm)
h5write(hdfname,"/data/grid/end",endm)
h5write(hdfname,"/data/grid/discretization",dxm)
h5writeatt(hdfname,'/data/','time', t);
h5writeatt(hdfname,'/data/','oxidizer', 'lowflux');
h5writeatt(hdfname,'/data/','personnel', 'EKI,KR');
h5writeatt(hdfname,'/data/','exp date', 'Feb 13');

cd ..
cd('Images')



%%
minT = min(Temp(Temp > 300));
maxT = max(Temp(Temp < 3500));
abcT = Temp(Temp > 300 & Temp < 3500);
meanT = mean(abcT);
minfv = min(sootFrac(sootFrac > 0));
maxfv = max(sootFrac(sootFrac < 1.15e-4));
abcfv = sootFrac(sootFrac > 0 & sootFrac < 1.15e-4);
meanfv = mean(abcfv);

% cd ..
% cd('Temperature22')
% save(append(filename(1:end-4),"mat"),'flameTemp');
% cd ..
% cd('Images')
%
% cd ..
% cd('fv22')
% save(append(filename(1:end-4),"soot.mat"),'fv');
% cd ..
% cd('Images')

% cd ..
% cd('HeatFlux22')
% save(append(filename(1:end-4),"mat"),'flameTemp');
% save(append(filename(1:end-4),"soot.mat"),'fv');
% cd ..
end

