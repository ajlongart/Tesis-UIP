%% Diseño de Filtro FIR Pasa Bajo Código SunFlicker
%Armando Longart 10-10844
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
imtool close all;  % Close all imtool figures.
clear;  % Erase all existing variables.

%% Parametros filtro Versión I (Tomando en cuenta el video)
vidObj = VideoReader('Sunlight_Reflection2_2.mp4');
get(vidObj) 
nFrames = vidObj.NumberOfFrames;    % Determine how many frames there are.
width = vidObj.Width;               % get image width
height = vidObj.Height;             % get image height
FPS = vidObj.FrameRate;
time = vidObj.Duration;
Fp  = 0.5;      % .5 Hz passband-edge frequency
Fst = 2;        % 20 Hz stop-edge frequency
Ap  = 1;        % Corresponds to 1 dB peak-to-peak ripple
Ast = 80;       % Corresponds to 60 dB stopband attenuation
N = 6;

%Filtro (usando FIR1)
Wn = 0.02;  %0.6/(FPS/2)
B = fir1(N,Wn,'low');
fvtool(B,'Fs',FPS,'Color','White')

%Filtro (usando equiripple lowpass fir)
eqnum = fdesign.lowpass('N,Fp,Fst',N,Fp,Fst,FPS);
f1 = design(eqnum, 'equiripple');    %Discrete-Time FIR Filter (real)
f1Num = f1.Numerator;   %Para filtros FIR (equiripple y kaiserwin)
info(f1)
fvtool(f1,'Fs',FPS,'Color','White')

%% Imagenes Nuevas para el Filtro
current = double(zeros(height,width));
previous1 = double(zeros(height,width));
previous2 = double(zeros(height,width));
previous3 = double(zeros(height,width));
previous4 = double(zeros(height,width));
previous5 = double(zeros(height,width));
previous6 = double(zeros(height,width));
salida = double(zeros(height,width));

%image8 = image1*B(1)+image2*B(2)+image3*B(3)+image4*B(4)+image5*B(5)+image6*B(6)+image7*B(7);


pixel_y = zeros(1,nFrames);
freqP_y = 1:nFrames;

for iFrame = 1:nFrames
    frame = read(vidObj,iFrame); % get one RGB image
    
    %Conversión del frame a YCbCr (Luminancia)
    ycbcr = rgb2ycbcr(frame);
    canal_y = ycbcr(:,:,1);
    canal_y = double(canal_y);
    %imshow(canal_y);
    
    %Para calcular la FFT de un pixel [Para obtener la frecuencia de corte] (Ahora sí)
    pixel_y(iFrame) = canal_y((3*height)/4,(3*width)/4);
    NFFT = 2^nextpow2(nFrames); % Next power of 2 from length of y
    modP_y = fft(pixel_y,NFFT)/length(pixel_y); 
    
    ejet = (0:length(pixel_y)-1)*time/length(pixel_y);
    ejef = FPS*linspace(0,1,NFFT/2+1);

%     P_y = fft(pixel_y)/length(pixel_y);
%     modP_y = abs(P_y);
%     ejef = (0:length(P_y)-1)*time/length(P_y);   % ejef = 0:fs/length(y):fs/N;
%     plot(ejef,modP_y);
    
%     current(iFrame)=canal_y(:,:,iFrame);
%     previous1(iFrame)=canal_y(:,:,iFrame);
%     previous2(iFrame)=canal_y(:,:,iFrame);
%     previous3(iFrame)=canal_y(:,:,iFrame);
%     previous4(iFrame)=canal_y(:,:,iFrame);
%     previous5(iFrame)=canal_y(:,:,iFrame);
%     previous6(iFrame)=canal_y(:,:,iFrame);
    
    %"Implementación del Filtro"
     for i = 1:height-1
        for j = 1:width-1
            salida(i,j) = canal_y(i,j)*(B(1)+B(2)+B(3)+B(4)+B(5)+B(6)+B(7));
            %salida(i,j) = B(1)*current(i,j)+B(2)*previous1(i,j)+B(3)*previous2(i,j)+B(4)*previous3(i,j)+B(5)*previous4(i,j)+B(6)*previous5(i,j)+B(7)*previous6(i,j);
        end
    end

    %salida = canal_y.*(B(1)+B(2)+B(3)+B(4)+B(5)+B(6)+B(7));
    %salida = canal_y.*(image1*B(1)+image2*B(2)+image3*B(3)+image4*B(4)+image5*B(5)+image6*B(6)+image7*B(7));
    %salida = convn(canal_y, image8, 'full');
    %salida(i,j) = conv2(canal_y(i,j), image8(i,j), 'same');                %https://stackoverflow.com/questions/20025604/applying-a-temporal-gaussian-filter-to-a-series-of-images
    
    imshow(salida,[]);
    %canal_y = double(canal_y);
    %imshow(canal_y,[]);
end

%% Plot de Resultados de FFT
figure
plot(ejet,pixel_y)  %Tiempo
figure
plot(ejef,2*abs(modP_y(1:NFFT/2+1))) %Frecuencia
