%% Diseño de Filtro FIR Pasa Bajo Código SunFlicker
%Armando Longart 10-10844
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
imtool close all;  % Close all imtool figures.
clear;  % Erase all existing variables.

%% Parametros filtro
%FIR Versión I
vidObj = VideoReader('Sunlight_Reflection2_2.mp4');
get(vidObj) 
nFrames = vidObj.NumberOfFrames;    % Determine how many frames there are.
width = vidObj.Width;               % get image width
height = vidObj.Height;             % get image height
FPS = vidObj.FrameRate;
time = vidObj.Duration;
Fp  = 0.5;        % 1 Hz passband-edge frequency
Fst = 2;        % 20 Hz stop-edge frequency
Ap  = 1;        % Corresponds to 1 dB peak-to-peak ripple
Ast = 80;       % Corresponds to 60 dB stopband attenuation
N = 6;

Wn = 0.02;  %0.6/(FPS/2)    1) 0.02; 2) 

B = fir1(N,Wn,'low');
fvtool(B,'Fs',FPS,'Color','White')

eqnum = fdesign.lowpass('N,Fp,Fst',N,Fp,Fst,FPS);
f1 = design(eqnum, 'equiripple');    %Discrete-Time FIR Filter (real)
f1Num = f1.Numerator;   %Para filtros FIR (equiripple y kaiserwin)
info(f1)

LPF = fdesign.lowpass('N,F3dB',N,Wn,FPS);
f2 = design(LPF, 'butter');    %Discrete-Time FIR Filter (real)
f2Num = f2.ScaleValues;   %Para filtros IIR (Butterworth)
info(f2)

fvtool(f2,'Fs',FPS,'Color','White')

%% Imagenes Nuevas para el Filtro
current = double(zeros(height,width));
previous1 = double(zeros(height,width));
previous2 = double(zeros(height,width));
previous3 = double(zeros(height,width));
previous4 = double(zeros(height,width));
previous5 = double(zeros(height,width));
previous6 = double(zeros(height,width));
salida = double(zeros(height,width));

pixel_y = zeros(1,nFrames);
pixel_salida = zeros(1,nFrames);

for iFrame = 1:nFrames
    frame = read(vidObj,iFrame); % get one RGB image
    
    grayImage = rgb2gray(frame);
    ycbcr = rgb2ycbcr(frame);
    canal_y = ycbcr(:,:,1);
    canal_y = double(canal_y);
    current = canal_y;
    
    pixel_y(iFrame) = canal_y((height)/2,(3*width)/4);
    
    ejet = (0:length(pixel_y)-1)*time/length(pixel_y);
    
    NFFT = 2^nextpow2(nFrames); % Next power of 2 from length of y
    modP_y = fft(pixel_y,NFFT)/length(pixel_y);
    ejef = FPS*linspace(0,1,NFFT/2+1);

    salida = B(1).*previous1+B(2).*previous2+B(3).*previous3+B(4).*previous4+B(5).*previous5+B(6).*previous6+B(7).*current;
    previous1 = previous2;
    previous2 = previous3;
    previous3 = previous4;
    previous4 = previous5;
    previous5 = previous6;
    previous6 = current;
    
    imshow(salida,[]);
    %canal_y = double(canal_y);
    %imshow(canal_y,[]);
    
    pixel_salida(iFrame) = salida((height)/2,(3*width)/4);
    
    ejets = (0:length(pixel_salida)-1)*time/length(pixel_salida);
    
    NFFT = 2^nextpow2(nFrames); % Next power of 2 from length of y
    mod_salida = fft(pixel_salida,NFFT)/length(pixel_salida);
    ejefs = FPS*linspace(0,1,NFFT/2+1);
end

%% Plot de Resultados
figure
plot(ejet,pixel_y)
title('Luminancia para pixel p(x)');
figure
plot(ejef,2*abs(modP_y(1:NFFT/2+1)))
title('FFT del pixel p(x)');

figure
plot(ejets,pixel_salida)
title('Luminancia para pixel Mod p(x)');
figure
plot(ejefs,2*abs(mod_salida(1:NFFT/2+1)))
title('FFT del pixel Mod p(x)');
