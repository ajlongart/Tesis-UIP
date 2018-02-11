%% Diseño de Filtro FIR Pasa Bajo Código SunFlicker
%Armando Longart 10-10844
%ajzlongart@gmail.com
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
imtool close all;  % Close all imtool figures.
clear;  % Erase all existing variables.

%% Parametros filtro
%FIR Versión I
vidObj = VideoReader('Sunlight_Reflection1_1.mp4');
get(vidObj) 
nFrames = vidObj.NumberOfFrames;    % Determine how many frames there are.
width = vidObj.Width;               % get image width
height = vidObj.Height;             % get image height
FPS = vidObj.FrameRate;
time = vidObj.Duration;
Fp  = 0.5;        % 0.5 Hz passband-edge frequency
Fst = 2;        % 2 Hz stop-edge frequency
Ap  = 1;        % Corresponds to 1 dB peak-to-peak ripple
Ast = 80;       % Corresponds to 80 dB stopband attenuation
N = 10;

Wn = 0.031;  %0.6/(FPS/2)    Frecuencia de corte normalizada

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
previous7 = double(zeros(height,width));
previous8 = double(zeros(height,width));
previous9 = double(zeros(height,width));
previous10 = double(zeros(height,width));
salida = double(zeros(height,width));

pixel_y1 = zeros(1,nFrames);
pixel_y2 = zeros(1,nFrames);
pixel_y3 = zeros(1,nFrames);
pixel_y4 = zeros(1,nFrames);
pixel_y5 = zeros(1,nFrames);

pixel_salida1 = zeros(1,nFrames);
pixel_salida2 = zeros(1,nFrames);
pixel_salida3 = zeros(1,nFrames);
pixel_salida4 = zeros(1,nFrames);
pixel_salida5 = zeros(1,nFrames);

for iFrame = 1:nFrames
    frame = read(vidObj,iFrame); % get one RGB image
    
    grayImage = rgb2gray(frame);
    ycbcr = rgb2ycbcr(frame);
    canal_y = ycbcr(:,:,1);
    canal_y = double(canal_y);
    %imshow(canal_y);
    current = canal_y;
    
    pixel_y1(iFrame) = canal_y((height)/2,(width)/2);
    pixel_y2(iFrame) = canal_y((height)/4,(width)/4);
    pixel_y3(iFrame) = canal_y((3*height)/4,(3*width)/4);
    pixel_y4(iFrame) = canal_y((height)/6,(width)/8);
    pixel_y5(iFrame) = canal_y((5*height)/6,(4*width)/5);
    
    ejet = (0:length(pixel_y1)-1)*time/length(pixel_y1);
    
    NFFT = 2^nextpow2(nFrames); % Next power of 2 from length of y
    modP_y1 = fft(pixel_y1,NFFT)/length(pixel_y1);
    modP_y2 = fft(pixel_y2,NFFT)/length(pixel_y3);
    modP_y3 = fft(pixel_y3,NFFT)/length(pixel_y3);
    modP_y4 = fft(pixel_y4,NFFT)/length(pixel_y4);
    modP_y5 = fft(pixel_y5,NFFT)/length(pixel_y5);
    ejef = FPS*linspace(0,1,NFFT/2+1);

%     salida = f2Num(1).*previous1+f2Num(2).*previous2+f2Num(3).*previous3+f2Num(4).*previous4+f2Num(5).*previous5+f2Num(6).*current;
%     previous1 = previous2;
%     previous2 = previous3;
%     previous3 = previous4;
%     previous4 = previous5;
%     previous5 = current;
    
    salida = B(1).*previous1+B(2).*previous2+B(3).*previous3+B(4).*previous4+B(5).*previous5+B(6).*current.*previous5+B(6).*previous6+B(7).*previous7+B(8).*previous8+B(9).*previous9+B(10).*previous10+B(11).*current;
    previous1 = previous2;
    previous2 = previous3;
    previous3 = previous4;
    previous4 = previous5;
    previous5 = previous6;
    previous6 = previous7;
    previous7 = previous8;
    previous8 = previous9;
    previous9 = previous10;
    previous10 = current;
    
    imshow(salida,[]);

    pixel_salida1(iFrame) = salida((height)/2,(width)/2);
    pixel_salida2(iFrame) = salida((height)/4,(width)/4);
    pixel_salida3(iFrame) = salida((3*height)/4,(3*width)/4);
    pixel_salida4(iFrame) = salida((height)/6,(width)/8);
    pixel_salida5(iFrame) = salida((5*height)/6,(4*width)/5);
    
    ejets = (0:length(pixel_salida1)-1)*time/length(pixel_salida1);
    
    NFFT = 2^nextpow2(nFrames); % Next power of 2 from length of y
    mod_salida1 = fft(pixel_salida1,NFFT)/length(pixel_salida1);
    mod_salida2 = fft(pixel_salida2,NFFT)/length(pixel_salida2);
    mod_salida3 = fft(pixel_salida3,NFFT)/length(pixel_salida3);
    mod_salida4 = fft(pixel_salida4,NFFT)/length(pixel_salida4);
    mod_salida5 = fft(pixel_salida5,NFFT)/length(pixel_salida5);
    ejefs = FPS*linspace(0,1,NFFT/2+1);
    
    % Resta de Imagenes
    resta = salida-canal_y;
    restafil = resta-salida;
    filt2 = or(salida,resta);
    %imshow(resta,[]);

end

%% Plot de Resultados
figure
plot(ejet,pixel_y1,ejet,pixel_y2,ejet,pixel_y3,ejet,pixel_y4,ejet,pixel_y5)
hleg1 = legend('pixel_y1','pixel_y2','pixel_y3','pixel_y4','pixel_y5');
axis([0,time,0,255])
grid on
title('Luminancia para pixel p(x)');
xlabel('tiempo (s)');
ylabel('Luminancia');
figure
plot(ejef,2*abs(modP_y1(1:NFFT/2+1)),ejef,2*abs(modP_y2(1:NFFT/2+1)),ejef,2*abs(modP_y3(1:NFFT/2+1)),ejef,2*abs(modP_y4(1:NFFT/2+1)),ejef,2*abs(modP_y5(1:NFFT/2+1)))
hleg2 = legend('FFT_y1','FFT_y2','FFT_y3','FFT_y4','FFT_y5');
grid on
title('FFT del pixel p(x)');
xlabel('frecuencia (Hz)');
ylabel('Luminancia');

figure
plot(ejets,pixel_salida1,ejets,pixel_salida2,ejets,pixel_salida3,ejets,pixel_salida4,ejets,pixel_salida5)
hleg3 = legend('pixel_salida1','pixel_salida2','pixel_salida3','pixel_salida4','pixel_salida5');
axis([0,time,0,255])
grid on
title('Luminancia para pixel Mod p(x)');
xlabel('tiempo (s)');
ylabel('Luminancia');
figure
plot(ejefs,2*abs(mod_salida1(1:NFFT/2+1)),ejefs,2*abs(mod_salida2(1:NFFT/2+1)),ejefs,2*abs(mod_salida3(1:NFFT/2+1)),ejefs,2*abs(mod_salida4(1:NFFT/2+1)),ejefs,2*abs(mod_salida5(1:NFFT/2+1)))
hleg4 = legend('FFT_salida1','FFT_salida2','FFT_salida3','FFT_salida4','FFT_salida5');
grid on
title('FFT del pixel Mod p(x)');
xlabel('frecuencia (Hz)');
ylabel('Luminancia');
