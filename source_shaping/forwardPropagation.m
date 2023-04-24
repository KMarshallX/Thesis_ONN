clc;clearvars;close all;
%%
load("test.mat");
load("testMasks.mat");
%% View one mask
idx = 6;
figure;
imagesc(squeeze(angle(MASKS(idx,:,:))));
colorbar;
%% Parameters initialization
%Centre Wavelength
lambda = 1565e-9;
%Phase mask pixel pitch
pixelSize = 9.2e-6;
%Plane spacing
planeSpacing = 50.06e-3;
%How far is the input SMF array from the first plane?
arrayDistToFirstPlane = 36.34e-3;
%Total number of planes
planeCount = 7;
%Pixel counts of the masks and simulation in x and y dimensions
Nx = 256;
Ny = 256;
%number of total passes to run for
iterationCount = 1000;
%Display graphs of fields/masks every N passes
graphIterations = 10;
%Mode-field diameter (MFD) of input Gaussian beams
MFDin = 864e-6;

%SIMULATION CONSTRAINTS
%Angle-space filter. Only propagate angles less than kSpaceFilter*(the
%maximum angle supported by the simulation. Which would in turn be given by
%the pixel size)
%e.g. 0.5 means only half the possible angle-space is used. Limiting angle
%space encourages solutions which are high-bandwidth, have low scatter,
%smooth and don't accidently wrap-around the edge of the simulation.
kSpaceFilter = 1000;

%A small offset that is added to the mask just before the phase-only is
%taken. This discourages the simulation from phase-matching low-intensity
%parts of the field, and encourages solutions which are higher-bandwidth,
%smoother and with less scatter. The Nx.*Ny.*modeCount normalization tries
%to keep the value consistent even if the resolution or number of modes is
%changed.
maskOffset = sqrt(1e-3./(Nx.*Ny));
%% Setup mask Cartesian co-ordinates
%0.5 pixel offset makes the problem symmetric in x and y
X = ((1:Ny)-(Ny./2+0.5)).*pixelSize;
Y = ((1:Nx)-(Nx./2+0.5)).*pixelSize;
[X, Y] = meshgrid(X,Y);
%Convert to polar-coordinates and rotate by 45degrees
[TH, R] = cart2pol(X,Y);
[X0, Y0] = pol2cart(TH-pi/4,R);
%array specifing the z-axis (offset to the first plane)
Z = ones(size(X)).*arrayDistToFirstPlane;
% Initialize a gaussian beam
[SPOT, TOTAL] = singleGaussianMode(Z, X, Y, MFDin, lambda); 
%Allocate all the fields (both directions, every plane, every mode, pixels
%x ,pixels y)
FIELDS = zeros(planeCount,Nx,Ny,'single');

%% free-space propagation
%The transfer function of free-space. This is used to propagate from
%plane-to-plane
%size(lambda)
H0 = transferFunctionOfFreeSpace(X,Y,planeSpacing,lambda);
%Filter the transfer function. Removing any k-components higher than
%kSpaceFilter*k_max.
maxR = max(max(R));
H = H0.*(R<(kSpaceFilter.*maxR));

%Initialise fields
%Put the spots as the field in the first plane travelling forward
FIELDS(1,:,:) = SPOT;

%% Setup forward propagation
%Index used to pick off the FIELD in this direction
directionIdx = 1;
%Transfer function of free-space (H)
h = H; % test , original: h = H;

for planeIdx=1:planeCount
    
    MASK = exp(-1i.*angle(squeeze(MASKS(planeIdx,:,:))));
    %Get the field of this mode in this plane
    field = squeeze(FIELDS(planeIdx,:,:));
    %Apply the mask
    field = field.*MASK;
    %Propagate it to the next plane
    field = propagate(field,h);
    %Store the result
    FIELDS(planeIdx+1,:,:) = field;
end

%% Plot the intensity patterns on each plane
figure(1);
for planeIdx = 1:planeCount
    total = squeeze(abs(FIELDS(planeIdx,:,:)).^2);
    subplot(2,4,planeIdx);
    imagesc(total);
    colorbar;
end

%% Plot each phase mask
figure(2);
for planeIdx = 1:planeCount
    phaseMask = squeeze(angle(MASKS(planeIdx,:,:)));
    subplot(2,4,planeIdx);
    imagesc(phaseMask);
    colorbar;
end

