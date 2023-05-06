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

%% 
fieldIn = conj(squeeze(FIELDS(1,planeCount,:,:)));
% apply phase mask
fieldIn = fieldIn.*squeeze(exp(1i.*angle(MASKS(planeCount,:,:))));
norm_fieldIn = (fieldIn - min(min(fieldIn))) ./ (max(max(fieldIn)) - min(min(fieldIn)));

figure;
imagesc(abs(norm_fieldIn));
colorbar;

figure;
imagesc(abs(fieldOut));
colorbar;

norm_OI = abs(sum(sum(norm_fieldIn.*fieldOut)));

numerator = abs(sum(sum(norm_fieldIn .* fieldOut))).^2;
denominator = sum(sum(abs(norm_fieldIn).^2)) .* sum(sum(abs(fieldOut).^2));
OI = numerator ./ denominator;
