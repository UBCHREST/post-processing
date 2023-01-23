clear all; close all; clc;
%% RAY TRACING KNOWING THE TEMPERATURE AND SOOT VOLUME FRACTION AT EACH LOCATION IN 3D SPACE
% Z = 1 IS THE LOCATION OF PMMA IN BOTH THE PROPERTIES MATRICES
% X IS THE WIDTH OF PMMA SAMPLE
% Y IS THE HEIGHT OF PMMA SAMPLE
% flameTempFinal = load('71Scale5LinearTemp.mat');
flameTempFinal = load('Img53.mat');
flameTempFinal = flameTempFinal.flameTemp;

% sootFractionFinal = load('71Scale5LinearSoot.mat');
sootFractionFinal = load('Img53soot.mat');
sootFractionFinal = sootFractionFinal.fv;

% Setting the azimuth angle from 0 to 2pi with pi/4 intervals
phi = linspace(0, 360, 121).*pi/180;
deltaPhi = phi(2) - phi(1);

% Setting the polar angle from 0 to pi/2 with pi/10 intervals
theta = linspace(0, 90, 31).*pi/180;
deltaTheta = theta(2) - theta(1);


% Maximum diagonal distance in bounding box
[sizeX, sizeY, sizeZ] = size(flameTempFinal);
% Fuel X, Y, Z coordinates

pmmaX = linspace(20, 368, 80); pmmaY = 13; pmmaZ = linspace(2, 19, 9.5);

% Create 12 triangles (2 each on the face of the bounding box)
% Triangle = createTriangle(boxFlame(1,1)-65, boxFlame(1,2)-30, pmmaZ, boxFlame(2,1)-60, boxFlame(2,2)-30, sizeZ);
Triangle = createTriangle(pmmaX(1), pmmaY, pmmaZ(1), sizeX, sizeY, sizeZ);

% Setting ds for intensity integration
ds = 0.5;

% Setting the ray length 
rho = 20;

% Setting the scale factors in front view and side view
scaleFront = pmmaX(2) - pmmaX(1);                                                      % cells per mm
scaleSide  = pmmaZ(2) - pmmaZ(1);                                                        % cells per mm
area = scaleSide*scaleFront;

% Optical constant for setup
C_green = 4.24;

% Initialize the total Intensity at each cell on PMMA surface
totalIntensity = zeros(length(pmmaX), length(pmmaZ));

% Create vector between PMMA and farthest point of ray to get maximum
% distance
tic;
% profile on
for i = 1:length(pmmaX)
    for j = 1:length(pmmaZ)
        for n = 1:length(phi)
            for m = 1:length(theta)
                % Compute the X, Y, Z coordinates of Ray End point
                rayX = pmmaX(i) + (rho*sin(theta(m))*cos(phi(n)));
                rayZ = pmmaZ(j) + (rho*sin(theta(m))*sin(phi(n)));
                rayY = pmmaY + (rho*cos(theta(m)));
                
                % Create a vector between the PMMA location and ray end
                % point
                [vecX, vecY, vecZ, vecMag] = createVector(pmmaX(i), pmmaY, pmmaZ(j), rayX, rayY, rayZ);
                % Computing the unit vectors ex, ey, ez of ray to enable
                % marching
                ex = vecX/vecMag;
                ey = vecY/vecMag;
                ez = vecZ/vecMag;
                
                % Origin of ray is stored as origin array
                origin = [pmmaX(i), pmmaY, pmmaZ(j)];
                % Direction vector stored as direction array 
                direction = [ex, ey, ez];
                boundingBox = [pmmaX(i), pmmaY, pmmaZ(j)];
                
                intersectionPoint = intersectionPointRayTriangle(origin, direction, Triangle, boundingBox);
                
                % Marching along the ray with unit distance ds
                % Initialize the initial new X, Y, Z coordinates with ray
                % triangle intersection point
                newX = intersectionPoint(1); newY = intersectionPoint(2); newZ = intersectionPoint(3);
                % Calculating the length of new ray
                [pointX, pointY, pointZ, lengthRay] = createVector(newX, newY, newZ, origin(1), origin(2), origin(3));
                % Initialize old intensity and total path intensity
                intensityOld = 0; intensityPath = 0;
                intensityOld2 = 0; intensityPath2 = 0;
                % While the current X, Y, Z coordinates are >= PMMA surface
%                 newLength = lengthRay;
                currentX = 2; currentY = 2; currentZ = 2;
                rayMarching = lengthRay:-ds:1;
                % Computing sine and cosine theta for current theta
                cosThetaTemp = cos(theta(m));
                sinThetaTemp = sin(theta(m));
%                 for stepping = 1:length(rayMarching)
                while (lengthRay >= 2 && currentX > 1 && currentY > 1 && currentZ > 1 && currentX < sizeX && currentY < sizeY && currentZ < sizeZ)
                    % March first ds and compute new X, Y, Z coordinates
                    currentX = newX - ex*ds;
                    currentY = newY - ey*ds;
                    currentZ = newZ - ez*ds; 
                    lengthRay = lengthRay - ds;
                    % Round the current X, Y, Z coordinates to nearest
                    % integer index
                    if (currentX < 1 )
                        roundX = ceil(abs(currentX));
                    else
                        roundX = round(abs(currentX));
                    end
                    if (currentY < 1)
                        roundY = ceil(abs(currentY));
                    else
                        roundY = round(abs(currentY)); 
                    end
                    if (currentZ < 1)
                        roundZ = ceil(abs(currentZ));
                    else
                        roundZ = round(abs(currentZ));
                    end
                    
                    currentTemperature = flameTempFinal(roundX, roundY, roundZ);
                    currentSoot = sootFractionFinal(roundX, roundY, roundZ);
                    % Compute extinction coefficient kappa =
                    % C*fv/lambda
                    kappaP = C_green*currentSoot/650e-9;
                    % Exponential function
                    expoFunc = exp(-kappaP*(ds*1e-3*cos(theta(m))/(scaleSide)));
                    % Emissivity calculation
                    epsilon = 1 - expoFunc;
                    % Blackbody intensity*epsilon
                    intensityBB = (intensity_flame(epsilon, currentTemperature));
                    % Current step intensity by RTE analytical
                    % integration
                    intensityNew = intensityBB + intensityOld*expoFunc;
                    % Sum up the total intensity along the ray
                    intensityPath = intensityPath + intensityNew;
                    % Update the old intensity
                    intensityOld = intensityNew;
                    % Update the new X, Y, Z coordinates for next
                    % marching step
                    newX = currentX;
                    newY = currentY;
                    newZ = currentZ;
                end
                if (abs(intensityPath) > 0)
                    rayIntensity = abs(intensityPath)*cosThetaTemp*sinThetaTemp*deltaPhi*deltaTheta;
                else
                    rayIntensity = 0;
                end
                totalIntensity(i,j) = totalIntensity(i,j) + rayIntensity;
            end
        end
    end
end

% Converting the total intensity to kW/m2
totalIntensity2 = totalIntensity./(area);
toc;

%%
figure1 = figure('Color',[1 1 1],'OuterPosition',[10 50 800 400]);
colormap(hot);

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

% Create contour
contourf(totalIntensity2');

box(axes1,'on');
% axis(axes1,'tight');
daspect([1 1 1]);
% Set the remaining axes properties
set(axes1,...
    'FontSize',20,'LineWidth',2.5,'XTick',...
    [1 10 20 30 40 50 60 70 80],'XTickLabel',...
    {'0','10','20','30','40','50','60','70','80'},'YTick',[1 5 8],'YTickLabel',...
    {'0','4.5','9.5'});

h = colorbar('northoutside', 'FontSize', 20);
title(h, '$\dot{q}^{\prime \prime} [W/m^2]$', 'FontSize', 20, 'Interpreter', 'latex');
caxis(([0 100]))



% Create textbox
annotation(figure1,'textbox',...
    [0.140139139139139 0.112469437652812 0.36836936936937 0.12958435207824],...
    'String','Flow Direction',...
    'FontWeight','bold',...
    'FontSize',25,...
    'FitBoxToText','off',...
    'EdgeColor','none');

% Create arrow
annotation(figure1,'arrow',[0.449449449449449 0.520520520520521],...
    [0.156479217603912 0.156479217603912],...
    'Color',[0 0.447058823529412 0.741176470588235],...
    'LineWidth',4,...
    'HeadWidth',15,...
    'HeadLength',15);


figure(2);
cla; grid on; hold on;
set(gca,'FontName','Arial','fontsize',25)
plot(pmmaX/(scaleFront), centerlineHF, 'r-', 'LineWidth', 2);
xlabel('Wax length $[mm]$', 'Interpreter', 'latex', 'FontName', 'Arial', 'FontSize', 20);
ylabel('$ \dot{q}^{\prime \prime}~[W/m^2]$', 'Interpreter', 'latex', 'FontName', 'Arial', 'FontSize', 20);
set(gca,'XMinorTick','on','YMinorTick','on');
set(gca, 'XTick', 0:20:100);
box on
set(gcf,'Color',[1,1,1],'OuterPosition',[10 10 750 750]);
set(gca,'linewidth',2.5)
axis(gca,'square');
set(gca,'ticklength',2*get(gca,'ticklength'))
hold off
grid on;