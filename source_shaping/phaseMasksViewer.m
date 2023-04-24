%%
clearvars; clear all;

%%
load("test.mat");
load("testMasks.mat");
%% View one mask
idx = 1;
phaseVals = squeeze(angle(MASKS(idx,:,:)));
% convert to grayscale image
grayscalePhase = uint8(mod(((round(256.*((phaseVals+pi)./(2.*pi))))),256));
figure;
imagesc(grayscalePhase);
colormap(gray(256));
colorbar;

