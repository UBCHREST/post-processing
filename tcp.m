clear all; close all; clc;


filename = dir('Img*.tiff')
for i=1:length(filename)
        currFileName = filename(i).name;
        dotcp(currFileName)
end


function []=dotcp(filename)
%% Reading in image and creating a mask
image = imread(filename);
% image = imread('Img759.tiff');
I = image(50:end,25:380,:);
image=image(end-100:end,:,:);
figure(999);
h = imshow(image);
e = drawpolygon;
vertexX = size(image,1) - floor(e.Position(:,2)) + 1;
vertexY = floor(e.Position(:,1)) + 1;
BW = createMask(e, h);
complementBW = imcomplement(BW);
image = image.*uint8(complementBW);

    
%%    
dt = log(1/400);   %reference f/22
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
t = 1/5000 ; %Time between Frames
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
T2 = (c2.*(1/lambdaR - 1./lambdaG))./(log(E(:,2)./E(:,1)) + log((lambdaG/lambdaR)^6) + log(4.25/4.55));

%Temperature in Celcius
% T = T2 - 273;

Temp_matrix = zeros(size(image_red));
Temp_matrix = T2;
Temp = ((reshape(Temp_matrix, [size(image,1), size(image,2), 1])));
for k = 1:size(image,1)
    for m = 1:size(image,2)
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


figure1 = figure('Color',[1 1 1],'OuterPosition',[10 50 800 450]);
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
    'FontSize',25,'LineWidth',2.5,'XTick',...
    [1 51.875 103.75 155.625 207.5 259.375 311.25 363.125 414],'XTickLabel',...
    {'0','10','20','30','40','50','60','70','80'},'YTick',[]);
% Create colorbar
c = colorbar(axes1,'northoutside','Ticks',[1800 2100 2400 2700 3000 3300]);
title(c, 'T [K]');
hold(axes1,'all');
saveas(figure1,append(filename(1:end-4),"jpg"))

%% CONSTRUCTING A 3D BOX FOR FLAME
flameTemp = 298.*ones(size(image,1), size(image,2), 20);
for k = 2:19
    flameTemp(:,:,k) = Temp;
end

% Computing the soot volume fraction with path length as the width of the
% sample
pathlength = 9.5e-3;

C0_red = 4.25;

Ilambda = reshape(E(:,1), size(Temp));

sootFrac = zeros(size(Temp));

for i = 1:size(Temp,1)
    for j = 1:size(Temp,2)
        if (Temp(i,j) > 400)
            sootFrac(i,j) = (-lambdaR.*1e9./(C0_red.*pathlength))*log(1 - (Ilambda(i,j)*lambdaR^5*exp(c2/(lambdaR*Temp(i,j))))/c1);
        else
            sootFrac(i,j) = 0;
        end
    end
end
% sootFrac(sootFrac > 300e-6) = mean(sootFrac(sootFrac < 200e-6));

fv = zeros(size(flameTemp));
for k = 2:19
    fv(:,:,k) = sootFrac;
end

flameTemp = permute(flameTemp, [2, 1, 3]);
fv        = permute(fv, [2, 1, 3]);

c = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2];

figure1 = figure('Color',[1 1 1],'OuterPosition',[10 50 800 450]);
colormap(hot);

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create contour
contourf(log(sootFrac), 30)
patch(vertexY, vertexX, 'green', 'FaceAlpha', 0.5);
box(axes1,'on');
axis(axes1,'tight');
% Set the remaining axes properties
set(axes1,'CLim',[-16.1180956509583 -4.60517018598809],'DataAspectRatio',...
    [1 1 1],'FontSize',20,'LineWidth',2.5,'XTick',...
    [1 51.875 103.75 155.625 207.5 259.375 311.25 363.125 414],'XTickLabel',...
    {'0','10','20','30','40','50','60','70','80'},'YTick',[]);
% Create colorbar
h = colorbar(axes1,'northoutside',...
    'Ticks',[-16.1180956509583 -13.8155105579643 -11.5129254649702 -9.21034037197618 -6.90775527898214 -4.60517018598809],...
    'TickLabels',{'10^-1','10^0','10^1','10^2','10^3','10^4'});
title(h, 'fv [ppm]');


%%
minT = min(Temp(Temp > 300))
maxT = max(Temp(Temp < 3500))
abcT = Temp(Temp > 300 & Temp < 3500);
meanT = mean(abcT)
minfv = min(sootFrac(sootFrac > 0))
maxfv = max(sootFrac(sootFrac < 1.15e-4))
abcfv = sootFrac(sootFrac > 0 & sootFrac < 1.15e-4);
meanfv = mean(abcfv)

save(append(filename(1:end-4),"mat"),'flameTemp');
end

