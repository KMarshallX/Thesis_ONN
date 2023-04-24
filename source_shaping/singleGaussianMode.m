function [SPOT, TOTAL] = singleGaussianMode(z, X, Y, MFD, lambda)
%beam waist is half the MFD
w0 = MFD./2;
%Pixel dimensions
s = size(X);
Nx = s(1);
Ny = s(2);

%the fields for the single spot
SPOT = zeros(Nx,Ny,'single');
%Summary of the total intensity of all spots
TOTAL = zeros(Nx,Ny,'single');

%calculate the spot
x = X;
y = Y;

%CALCULATE A GAUSSIAN BEAM IN X,Y,Z
zr = pi.*w0.^2./lambda;
k = 2.*pi./lambda;
Wz = w0.*sqrt(1+(z./zr).^2);
phi = atan(z./zr);
Rz = z.*(1+(zr./z).^2);
Rz_ = 1./Rz;
Rz_(isinf(Rz_) | isnan(Rz_))=0;
R2 = (x.^2+y.^2);
Rzterm = R2.*Rz_./(2);
spot = (w0./Wz).*exp(-R2./Wz.^2).*exp(-1i.*(k.*z+k.*Rzterm-phi));

%Normalize the spot to unit intensity
norm = sqrt(sum(sum(sum(abs(spot).^2))));
if (norm>0)
    spot = spot./norm;
    TOTAL=TOTAL+abs(spot).^2;
end
SPOT = spot;

end