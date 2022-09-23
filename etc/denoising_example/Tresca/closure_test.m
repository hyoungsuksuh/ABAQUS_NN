% ------------------------------------------------------------- %
% Written by: Hyoung Suk Suh (Columbia University)              %
% Closure test in resampled data points                         %
% ------------------------------------------------------------- %

clear all; close all; clc

addpath util

% Load resampled datapoints from the set of local patches 
data_sample = csvread('./tresca_data_preprocess/data_resampled.csv',0,0);

p     = zeros(length(data_sample),1);
rho   = zeros(length(data_sample),1);
theta = zeros(length(data_sample),1);

for i = 1:length(data_sample)

    sigma1_pp = data_sample(i,1);
    sigma2_pp = data_sample(i,2);
    sigma3_pp = data_sample(i,3);
    
    rho(i)   = sqrt(sigma1_pp^2 + sigma2_pp^2);
    theta(i) = atan2(sigma2_pp, sigma1_pp);
    p(i)     = (1/sqrt(3))*sigma3_pp;
    
    if theta(i) < 0
        theta(i) = theta(i) + 2*pi;
    end
    
end

xx = rho.*cos(theta);
yy = rho.*sin(theta);


% Project x and y into a regular grid
eps = 1.5;
grid_xlen = 301;
grid_ylen = 301;

center_x = round(grid_xlen/2);
center_y = round(grid_ylen/2);

[grid_xcoord, grid_ycoord] ...
= meshgrid(linspace(1, grid_xlen, grid_xlen), linspace(0, grid_ylen, grid_ylen));

grid_xcoord = grid_xcoord - center_x;
grid_ycoord = grid_ycoord - center_y;

grid = logical(zeros(grid_xlen, grid_ylen));
for i = 1:length(xx)
    
    target_point_x = xx(i);
    target_point_y = yy(i);
    for ii = 1:grid_xlen
        for jj = 1:grid_ylen
            if sqrt((grid_xcoord(ii,jj) - target_point_x)^2 + (grid_ycoord(ii,jj) - target_point_y)^2) < eps 
                grid(ii,jj) = 1;
            end
            
        end
    end
end

figure;
imagesc(grid); 
colormap(bone);
colorbar; axis equal; axis off;


% Test if the loop is closed
J = regiongrowing(grid,center_x,center_y,0.2);
J = -2.*(J-0.5);

figure;
imagesc(J); 
colormap(jet);
colorbar; axis equal; axis off;


% Recover signed distance function
signed_dist = bwdist(grid);
signed_dist = J.*signed_dist;

figure;
imagesc(signed_dist);
colormap(jet); colorbar; axis equal; axis off;