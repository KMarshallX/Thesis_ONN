%%
clearvars; clc;
close all;

%% Load output image patterns
load("test.mat");

%% Parameters initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input mode field diameter: 864 um
% Output field size: 350 um -> digit size: 38*38 pixels
% Distance to the first plane: 36.34 mm
% Distance between planes: 50.06 mm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
iterationCount = 200;
%Display graphs of fields/masks every N passes
graphIterations = 5;
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
%Calculate all spots at positions (x,y) using the co-ordinate system
%(X,Y,Z), at the specified wavelength (lambda)
[SPOT, TOTAL] = singleGaussianMode(Z, X, Y, MFDin, lambda); 

%Print-out how much memory you're going to need for all the fields (every
%mode at every plane, in both directions).
memoryRequiredGB = (2.*planeCount.*Nx.*Ny.*8)./(1024.^3);
fprintf('This simulation requires over %3.3f GB of RAM\n',memoryRequiredGB);

%Allocate all the fields (both directions, every plane, every mode, pixels
%x ,pixels y)
FIELDS = zeros(2,planeCount,Nx,Ny,'single');

%Allocate all masks. Here set to blank phase, but these could be any
%initial state you wish. e.g. lens-like masks
MASKS = ones(planeCount,Nx,Ny,'single');

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
FIELDS(1,1,:,:) = SPOT;
%Put the required pattern as the field in the last plane travelling backward
FIELDS(2,planeCount,:,:) = squeeze(images(1,:,:));

%% Setup the initial fields at each plane in each direction...
%% Setup forward propagation
%Index used to pick off the FIELD in this direction
directionIdx = 1;
%Transfer function of free-space (H)
h = H; % test , original: h = H;

for planeIdx=1:(planeCount-1)
    %Conjugate of the mask in this plane. Whether to conjugate or not just
    %depends on how you've set up all the conjugates, fields and overlaps
    %throughout the simulation. Main thing is to be consistent throughout.
    MASK = exp(-1i.*angle(squeeze(MASKS(planeIdx,:,:))));
    
    %Get the field of this mode in this plane
    field = squeeze(FIELDS(directionIdx,planeIdx,:,:));
    %Apply the mask
    field = field.*MASK;
    %Propagate it to the next plane
    field = propagate(field,h);
    %Store the result
    FIELDS(directionIdx,planeIdx+1,:,:) = field;
end

%% Setup backwards field
%Index used to pick off the FIELD in this direction
directionIdx = 2;
%Travelling backwards so the transfer function of free-space is conjugate
%(-z). Again, what is and isn't conjugated just depends on the conventions
%you've chosen, but will have to be consistent throughout for it to work.
%h = conj(H0); %test
h = conj(H); % original
for planeIdx=planeCount:-1:2
    %The phase of the mask in this plane
    MASK = exp(1i.*angle(squeeze(MASKS(planeIdx,:,:))));
    
    field = squeeze(FIELDS(directionIdx,planeIdx,:,:));
    %Apply the mask
    field = field.*MASK;
    %Propagate backwards to the previous plane
    field = propagate(field,h);
    %Store the result
    FIELDS(directionIdx,planeIdx-1,:,:) = field;
    
end
%All the fields are initialised now. Technically we didn't need to setup
%the forward directionupdateMaskNew as we'll be starting from the first plane and 
%re-calculatinupdateMaskg that in the first iteration.

%% Time to iterate through and update the masks so attempt to phase-match all
%modes propagating in both directions.
pb = waitbar(0, 'Please wait...');
for i=1:iterationCount
    
    %Propagate from first plane to last plane
    % h = H0; % test
    h = H; % original
    directionIdx=1;
    for planeIdx=1:(planeCount-1)
        %Update the mask (see seperate script updateMask.m)
        updateMaskNew;
        %Take the conjugate phase of the mask in this plane
        MASK = exp(-1i.*angle(squeeze(MASKS(planeIdx,:,:))));
   
        field = squeeze(FIELDS(directionIdx,planeIdx,:,:));
        %Apply the mask
        field = field.*MASK;
        %Progagate to the next plane
        field = propagate(field,h);
        %Store the result
        FIELDS(directionIdx,planeIdx+1,:,:) = field;

    end
    
    %Propagate backwards from last plane to first plane
    %Propagating backwards so conjugate transfer function
    % h = conj(H0); %test
    h = conj(H); % original
    directionIdx=2;
    for planeIdx=planeCount:-1:2
        %Update the mask (see seperate script updateMask.m)
        updateMaskNew;

        %the phase of the mask for this plane
        MASK = exp(1i.*angle(squeeze(MASKS(planeIdx,:,:))));
        
        field = (squeeze(FIELDS(directionIdx,planeIdx,:,:)));
        %Apply the mask
        field = field.*(MASK);
        %Propagate backwards to previous plane
        field = propagate(field,h);
        %Store the result
        FIELDS(directionIdx,planeIdx-1,:,:) = (field);
 
    end
    %Graph the fields and masks every graphIterations passes
    if (mod(i,graphIterations)==0)
        newGraphFields;
    end

    waitbar(i/iterationCount, pb, sprintf('%d / %d', i, iterationCount));
end
close(pb)

%% Final error calculation
%Use the un-filtered transfer function of free-space (the entire
%angle-space)
h = H0;
%Propagate from the first plane to the last plane in the forward direction
directionIdx = 1;
for planeIdx=1:(planeCount-1)
    %Conjugate phase of the mask in this plane.
    MASK = exp(-1i.*angle(squeeze(MASKS(planeIdx,:,:))));
    % only one mode
    field = squeeze(FIELDS(directionIdx,planeIdx,:,:));
    %Apply mask
    field = field.*MASK;
    %Propagate to next plane
    field = propagate(field,h);
    %Store the result
    FIELDS(directionIdx,planeIdx+1,:,:) = field;
end

% overlap integral
fieldIn = squeeze(FIELDS(1,planeCount,:,:));
% % apply phase mask
% fieldIn = fieldIn.*squeeze(exp(1i.*angle(MASKS(planeCount,:,:))));
% normalization
fieldIn = fieldIn ./ sqrt(sum(sum(abs(fieldIn).^2)));
%get the field in the backward direction at the last plane
fieldOut = squeeze(FIELDS(2,planeCount,:,:));
% normalization
fieldOut = fieldOut ./ sqrt(sum(sum(abs(fieldOut).^2)));
%Calculate the overlap integral between the two (fieldIn already
%conjugated)
overlapVector = sum(sum(conj(fieldIn) .* fieldOut));
intensityOverlap = abs(overlapVector) .^ 2;% Result = 0.0828


% E1 = fieldIn;
% E2 = fieldOut;
% numerator = abs(sum(sum(E1 .* E2))).^2;
% denominator = sum(sum(abs(E1).^2)) .* sum(sum(abs(E2).^2));
% OI = numerator ./ denominator;  % OI = 0.2706

% Single img processing done
fprintf('Single img processing done!\n');
fprintf('The overlap integral of the two fields in the last plane is %.4f', intensityOverlap);
    
%%
% save("testMasks.mat", "MASKS")





