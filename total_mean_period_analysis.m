%% -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------%%
%% --  %% 程式輸入
% 此程式主要用來分析 馬達不同時期損壞特徵之變化
% 輸入每一時期分析data 隻數量 及 時期的總數
% 需要輸入馬達損壞之特徵頻率 
% 馬達名稱 每一時期收值之日期
% 檔案存取位置
% 如果sensor 讀取的data和實際data間有轉換公式 也需要輸入他們間之關係
% 輸入 sensor sampling rate 和 wpd 使用之 reampling rate
% 訊號分析之種類 :{1.原訊號 與 2.Envelop後之訊號}
% 讀取 data 之行列位置
% 欲抓取data 之初始時間和末了時間
% 自動抓點的 fspan 及欲分析之斜坡個數
% 輸入馬達名字
%--------------------------------------------------------------------------------------------------------------------%

close all;clc;clear all;
%% ------------------------------------------------------------------------------------------- 使用者分析輸入-------------------------------------------------------------------------------------------------------------
% 馬達名稱 每一時期收值之日期

motor = [1,6,11];
number = 2; % (number 1 = 一號機 ;number 2 = 六號機 ; number 3 = 十一號機)
motorname = num2str(motor(number));
% get data mode
mode = 1;
% 輸入 sensor sampling rate 和 wpd 使用之 reampling rate
samplerate = 25000;
resamplerate = 1280;    
level = 8;

%% 是否 resamplerate
resam_or_not = 0;
% 選擇畫什麼樣的圖
plotpici = 0;
plotpicv = 0;
plotpiciv = 0;
plotpictrend = 1;
plotpiciRE = 0;
plotime = [1 1 1 0 1 1 1];
% 自動抓點的 fspan 及欲分析之斜坡個數
harmonics = 55;
mechfalt_num = 0;
fspan = 1;
lp_cut_f = 12500;

% 訊號分析之種類 :{1.原訊號 與 2.Envelop後之訊號}
signal_type_select = 1;
signal_type= {'Normal','Envelope'};
date = {'8.1','8.15','9.5','10.9','11.13','12.11','1.15'};
period = size(date,2);

% 檔案存取位置
FolderName =['G:\小組雲端硬碟\華邦電子\電流特徵頻譜圖\',motorname,' 號機\',signal_type{signal_type_select}];   % Your destination folder

%%
x0=10;
y0=10;
width=900;
height=700;

startT = 0;
finalT = 10;

fband = [0 400];
a = 0;

Highband = [15000:1:25000;
            25000:1:35000;
            25000:1:35000];

T = 1/samplerate;
time = finalT-startT ;
t = startT+T :T :finalT;

number_wpt_fre = 3;


%% 1 6 11 馬達 各自的電流特徵頻率
if a == 0
mbrokefreq(1,:,1) = [53.2,52.8,53.6,26.7,79.7,183.825,134.575,161.114];
mbrokefreq(2,:,1) = [53.1,52.1,54.1,26.8,79.4,183.53,133.37,160.2];
mbrokefreq(3,:,1) = [53.5,52.9,54.1,26.9,80.1,184.03,135.77,164.759];
% 8/1 Second times 三台馬達的特徵頻率

mbrokefreq(1,:,2) = [53.5,52.9,54.1,26.9,80.1,184.618,135.182,161.822];
mbrokefreq(2,:,2) = [53.4,52.6,54.2,26.9,79.9,184.339,136.061,161.314];
mbrokefreq(3,:,2) = [53.8,53,54.6,27.1,80.5,184.739,136.461,165.896];
% 8/15 Third times 三台馬達的特徵頻率

mbrokefreq(1,:,3) = [53.1,52.5,53.7,26.7,79.5,184.232,134.168,160.607];
mbrokefreq(2,:,3) = [53.4,52.6,54.2,26.9,79.9,184.829,136.671,161.314];
mbrokefreq(3,:,3) = [53.9,53.3,54.5,27.1,80.7,185.329,136.871,166.426];
% 9/5 Third times 三台馬達的特徵頻率

mbrokefreq(1,:,4) = [53,52.6,53.4,26.6,79.4,183.132,134.068,160.236];
mbrokefreq(2,:,4) = [53.8,53,54.6,27.1,80.5,184.739,136.461,162.529];
mbrokefreq(3,:,4) = [53.6,53.2,54,26.9,80.3,184.539,136.261,165.696];
% 10/9 Forth times 三台馬達的特徵頻率

mbrokefreq(1,:,5) = [53.7,53.1,54.3,26.6,80,183.991,135.289,162.029];
mbrokefreq(2,:,5) = [54,53.2,54.8,27.2,80.8,186.104,136.296,163.136];
mbrokefreq(3,:,5) = [53.7,53.1,54.3,27,80.4,184.721,136.279,165.377];
% 11/13 Fifth times 三台馬達的特徵頻率

mbrokefreq(1,:,6) = [54.1,53.5,54.7,27.2,81,186.697,136.703,163.643];
mbrokefreq(2,:,6) = [54.1,53.5,54.7,27.2,81,186.697,136.703,163.643];
mbrokefreq(3,:,6) = [54,53.6,54.4,27.1,80.9,186.002,137.198,166.514];
% 12/11 sixth times 三台馬達的特徵頻率

mbrokefreq(1,:,7) = [54.1,53.5,54.7,27.2,81,186.697,136.703,163.643];
mbrokefreq(2,:,7) = [54,53.4,55,27.3,81.1,186.797,136.803,163.743];
mbrokefreq(3,:,7) = [54.2,53.4,55,27.3,81.1,186.202,137.398,166.714];

stringb = {'Fbl','Fbr','Fmisl','Fmisr','Fi','Fo','Fre'};

else
    
    
mbrokefreq(1,:,1) = [53.2,1704,1810,2023,2130,186.697,134.575,161.114];
mbrokefreq(2,:,1) = [53.1,2788, 2894, 3106, 3212,183.53,133.37,160.2];
mbrokefreq(3,:,1) = [53.5,2791, 2898, 3112, 3219,184.03,135.77,164.759];
% 8/1 Second times 三台馬達的特徵頻率

mbrokefreq(1,:,2) = [53.5,1712, 1819, 2033, 2140,186.697,135.182,161.822];
mbrokefreq(2,:,2) = [53.4,2789, 2893, 3107, 3214,184.339,136.061,161.314];
mbrokefreq(3,:,2) = [53.8,2790, 2898, 3113, 3220,184.739,136.461,165.896];
% 8/15 Third times 三台馬達的特徵頻率

mbrokefreq(1,:,3) = [53.1,1699, 1805, 2018, 2124,186.697,134.168,160.607];
mbrokefreq(2,:,3) = [53.4,2787, 2894 ,3107, 3214,184.829,136.671,161.314];
mbrokefreq(3,:,3) = [53.9,2790, 2897, 3113, 3221,185.329,136.871,166.426];
% 9/5 Third times 三台馬達的特徵頻率

mbrokefreq(1,:,4) = [53,1697,  1803, 2015, 2122,186.697,134.068,160.236];
mbrokefreq(2,:,4) = [53.8,2785, 2893, 3108, 3216,184.739,136.461,162.529];
mbrokefreq(3,:,4) = [53.6,2791, 2898, 3112, 3220,184.539,136.261,165.696];
% 10/9 Forth times 三台馬達的特徵頻率

mbrokefreq(1,:,5) = [53.7,1719, 1827, 2042, 2149,186.697,135.289,162.029];
mbrokefreq(2,:,5) = [54,2784, 2892, 3108, 3216,186.104,136.296,163.136];
mbrokefreq(3,:,5) = [53.7,2791, 2898, 3112, 3220,184.721,136.279,165.377];
% 11/13 Fifth times 三台馬達的特徵頻率

mbrokefreq(1,:,6) = [54.1,1732, 1840, 2057, 2160,186.697,136.703,163.643];
mbrokefreq(2,:,6) = [54.1,2784, 2892, 3108, 3216,186.697,136.703,163.643];
mbrokefreq(3,:,6) = [54,2789, 2897, 3113, 3221,186.002,137.198,166.514];
% 12/11 sixth times 三台馬達的特徵頻率

mbrokefreq(1,:,7) = [54.1,1730, 1838, 2054, 2160,186.697,136.703,163.643];
mbrokefreq(2,:,7) = [54,2785, 2892, 3108, 3216,186.797,136.803,163.743];
mbrokefreq(3,:,7) = [54.2,2788,2897,3114,3222,186.202,137.398,166.714];
% 1/15 seventh times 三台馬達的特徵頻率
    
 
stringb = {'HighFre1','HighFre2','HighFre3','HighFre4','Fi','Fo','Fre'};   
end

%% ------------------------------------------------------------------------------------------------主程式分析---------------------------------------------------------------------------------------------------------------
%% 讀取 馬達 data 
% 檔案讀取位置

switch mode
    case 1
[filename1, pathname1] = uigetfile(...
{'*.txt','Text Files (*.txt)';}, ...
 'Pick a File',...
 'MultiSelect', 'on');
name1 = strcat(pathname1,filename1);

[filename2 ,pathname2] = uigetfile(...
{'*.txt','Text Files (*.txt)';}, ...
 'Pick a File',...
 'MultiSelect', 'on');
name2 = strcat(pathname2,filename2);

[filename3 ,pathname3] = uigetfile(...
{'*.txt','Text Files (*.txt)';}, ...
 'Pick a File',...
 'MultiSelect', 'on');
name3 = strcat(pathname3,filename3);


    case 2
        for i=1:1:period
        inputfolder = ['D:\Data\',motorname,'\',date{i}];
        files = struct2cell(dir(fullfile(inputfolder, '*.txt')));
        name(i,:) = strcat(inputfolder,'\',{files{1,:}});
        end
        name1 = name(1,:);
        name2 = name(2,:);
        name3 = name(3,:);
end

amount = size(name,2);
d = cell(amount,period);


for i =1:1:period
    for j=1:1:amount
    d{j,i} = importdata(name{i,j},'\t',10);
    end
end


%% Power 電壓 電流 RMS 變數宣告
Power = zeros(amount,period);
RMSV = zeros(amount,period);
RMSI = zeros(amount,period);
RMSP = zeros(amount,period);


%% 
point_num = samplerate*startT+1:samplerate*finalT;

if resam_or_not == 1
    sam = resamplerate;
else
    sam = samplerate;
end
    
TotalI = zeros(amount,period,sam*time);
TotalV = zeros(amount,period,sam*time);
SumI = zeros(sam*time,1);
SumV = zeros(sam*time,1);
Average_I = zeros(sam*time,period);
Average_V = zeros(sam*time,period);
Average_P = zeros(sam*time,period);
  
Average_rms_V = zeros(1,period);
Average_rms_I = zeros(1,period);
Average_rms_P = zeros(1,period);

percent_thdV = zeros(period,1);
percent_thdI = zeros(period,1);
percent = percent_thdI./percent_thdV;

%% 電壓倍頻 損壞頻率 變數宣告
Fall_c = zeros(harmonics+7+mechfalt_num,2,amount);
Fall_v = zeros(harmonics+7+mechfalt_num,2,amount);
Fall_c_v = zeros(harmonics+7+mechfalt_num,2,amount);

Fault_c_bearing = zeros(3,amount);

Average_Fall_c = zeros(harmonics+7+mechfalt_num,2,period);
Average_Fall_c_v = zeros(harmonics+7+mechfalt_num,2,period);

Fault_c_wpd_coefficient = zeros(3,resamplerate*time,amount);
Fault_v_wpd_coefficient = zeros(3,resamplerate*time,amount);

Average_Fault_c_wpd_coefficient = zeros(3,resamplerate*time,period);
Average_Fault_v_wpd_coefficient = zeros(3,resamplerate*time,period);

Total_thd_dbV = 0;
Total_thd_dbI = 0;

thd_total = zeros(period,amount);


I_fft = zeros(samplerate*time,amount);
I_V_fft = zeros(samplerate*time,amount);

Average_I_fft = zeros(samplerate*time,period);
Average_I_V_fft = zeros(samplerate*time,period);

current_harmonics = zeros(harmonics-1,period);



dis = 1.25;
     
fspanvar1 = zeros(amount,period);
fspanvar2 = zeros(amount,period);
     
fspanvar3 = zeros(amount,period);
fspanvar4 = zeros(amount,period);
fspanvar5 = zeros(amount,period);
      
fspanmean3 = zeros(amount,period);
fspanmean4 = zeros(amount,period);
fspanmean5 = zeros(amount,period);
      
      
fspanvar_cv3 = zeros(amount,period);
fspanvar_cv4 = zeros(amount,period);
fspanvar_cv5 = zeros(amount,period);
      
fspankur1 =  zeros(amount,period);
fspankur3 = zeros(amount,period);
fspankur5 = zeros(amount,period);


      
mean_wpd_I = zeros(3,amount,period);
rms_wpd_I = zeros(3,amount,period);
std_wpd_I = zeros(3,amount,period);
mean_wpd_IV = zeros(3,amount,period);
rms_wpd_IV = zeros(3,amount,period);
std_wpd_IV = zeros(3,amount,period);
       
       
br_faltL = zeros(period,amount);
br_faltR = zeros(period,amount);
       
fmisL = zeros(period,amount);
fmisR = zeros(period,amount);
    
fmis3 = zeros(period,amount);
fmis5 = zeros(period,amount);
HighFreSum = zeros(period,amount);
        
foddharmonic = zeros(period,amount,floor((harmonics-1)/2));
        
cov_PV = zeros(period,amount);
crr_PV = zeros(period,amount);
AA = zeros(amount,period);
%% 電壓倍頻變數宣告
harmonicsV = zeros(period,amount);
HighFre = zeros(period,amount);

for i=1:1:period    
    for j=1:1:amount
        samplerate = 25000;

        if i<=3
        %% inital setting
    % 7.8.9 三相電壓  10.11.12 三相電流 華邦
        cloumnumv = [7,8,9] ;
        cloumnumc = [10,11,12] ;

        dataI1 = d{j,i}.data(:,cloumnumc(1));
        dataI2 = d{j,i}.data(:,cloumnumc(2));
        dataI3 = d{j,i}.data(:,cloumnumc(3));

        dataVA = d{j,i}.data(:,cloumnumv(1));
        dataVB = d{j,i}.data(:,cloumnumv(2));
        dataVC = d{j,i}.data(:,cloumnumv(3));

        I1 = dataI1(point_num);
        I2 = dataI2(point_num);
        I3 = dataI3(point_num);

        VA = dataVA(point_num);
        VB = dataVB(point_num);
        VC = dataVC(point_num);

        V1 = VA-VB;
        V2 = VB-VC; 
        V3 = VC-VA;
        else
        cloumnumv = [8,9,10] ;
        cloumnumc = [12,13,14] ;
        dataI1 = d{j,i}.data(:,cloumnumc(1));
        dataI2 = d{j,i}.data(:,cloumnumc(2));
        dataI3 = d{j,i}.data(:,cloumnumc(3));

        I1 = dataI1(point_num);
        I2 = dataI2(point_num);
        I3 = dataI3(point_num);

        dataVA = d{j,i}.data(:,cloumnumv(1));
        dataVB = d{j,i}.data(:,cloumnumv(2));
        dataVC = d{j,i}.data(:,cloumnumv(3));

        V1 = dataVA(point_num);
        V2 = dataVB(point_num);
        V3 = dataVC(point_num);
        end


        if resam_or_not == 1
        [p,q] = rat(resamplerate/samplerate);
        I1 = resample(I1((startT*samplerate+1):(finalT*samplerate)),p,q);
        V1 = resample(V1((startT*samplerate+1):(finalT*samplerate)),p,q);
        samplerate = resamplerate;
    %     I1 = downsample(I1,10,0);
    %     V1 = downsample(I1,10,0);
    %      samplerate = resamplerate;
        end

        Power(j,i) = sum(I1.*V1)/10;
        RMSV(j,i) = rms(V1);
        RMSI(j,i) = rms(I1);
        RMSP(j,i) = RMSV(j,i)*RMSI(j,i);
        SumI = SumI + I1 ;
        SumV = SumV + V1 ; 
        TotalI(j,i,:) = I1;
        TotalV(j,i,:) = V1;

        Ia = rms(I1);
        Ib = rms(I2);
        Ic = rms(I3);
        Im = (Ia+Ib+Ic)/3;
        Va = rms(V1);
        Vb = rms(V2);
        Vc = rms(V3);
        Vm = (Va+Vb+Vc)/3;

        unbalancedI(:,i,j) = [abs(Ia-Im)/Im;abs(Ib-Im)/Im;abs(Ic-Im)/Im];
        unbalancedV(:,i,j) = [abs(Va-Vm)/Vm;abs(Vb-Vm)/Vm;abs(Vc-Vm)/Vm];

        %% 計算電流電壓輸入之斜坡失真
        [thd_dbV,harmpowV,harmfreqV] = thd(V1,samplerate,50);
         [thd_dbI,harmpowI,harmfreqI] = thd(I1,samplerate,50);
         thd_total(i,j) =  100*(10^(thd_dbV/20));
        Total_thd_dbV = Total_thd_dbV + 100*(10^(thd_dbV/20));
        Total_thd_dbI = Total_thd_dbI + 100*(10^(thd_dbI/20));



    %     q = 100*(10^(thd_dbV/20)) 
    %% ------------------------------------------------------------------------------------------------------------------------------------找頻譜特徵  畫 FFT-----------------------------------------------------------------------------------------------------------


        [Fall_c(:,:,j) I_fft(:,j)] = plot_fft2(I1,samplerate, fspan ,mbrokefreq(number,1,i), mbrokefreq(number,2,i),mbrokefreq(number,3,i), mbrokefreq(number,4,i),... 
                       mbrokefreq(number,5,i), mbrokefreq(number,6,i), mbrokefreq(number,7,i), mbrokefreq(number,8,i), RMSP,date,motorname,signal_type{signal_type_select},harmonics,i,1,j,fband,plotpici,plotime,mechfalt_num,stringb); 


        [Fall_v(:,:,j) V_fft(:,:,j)] = plot_fft2(V1,samplerate, fspan ,mbrokefreq(number,1,i), mbrokefreq(number,2,i),mbrokefreq(number,3,i), mbrokefreq(number,4,i),... 
                        mbrokefreq(number,5,i), mbrokefreq(number,6,i), mbrokefreq(number,7,i), mbrokefreq(number,8,i), RMSP,date,motorname,signal_type{signal_type_select},harmonics,i,2,j,fband,plotpicv,plotime,mechfalt_num,stringb); 


%       [Fall_c_v(:,:,j) I_V_fft(:,j)] = plot_fft_cv(I1,V1,samplerate ,Fall_c(:,1,j),RMSP,date,motorname,signal_type{signal_type_select},harmonics,i,j,fband,plotpiciv,0.5,plotime,mechfalt_num);

        Fault_c_bearing(:,j) = [Fall_c(harmonics+5,1,j);Fall_c(harmonics+6,1,j);Fall_c(harmonics+7,1,j)];    

        [Fault_c_wpd_coefficient(:,:,j),freqbandi,wptreeI] = WPD_coefficient(I1,samplerate,startT,finalT,Fault_c_bearing,resamplerate,signal_type{signal_type_select},motorname,level);

        [Fault_v_wpd_coefficient(:,:,j),freqbandv,wptreeV] = WPD_coefficient(V1,samplerate,startT,finalT,Fault_c_bearing,resamplerate,signal_type{signal_type_select},motorname,level);


    
    
%% Notch Filter
        % fs = 25000;             %#sampling rate
        % f0 = mbrokefreq(number,1,i);                %#notch frequency
        % fn = fs/2;              %#Nyquist frequency
        % freqRatio = f0/fn;      %#ratio of notch freq. to Nyquist freq.
        % 
        % notchWidth = 0.3;       %#width of the notch
        % 
        % %Compute zeros
        % notchZeros = [exp( sqrt(-1)*pi*freqRatio ), exp( -sqrt(-1)*pi*freqRatio )];
        % 
        % %#Compute poles
        % notchPoles = (1-notchWidth) * notchZeros;
        % 
        % dd = poly( notchZeros ); %# Get moving average filter coefficients
        % cc = poly( notchPoles ); %# Get autoregressive filter coefficients
        % 
        % %#filter signal x
        % Sig = filter(dd,cc,I1);




  %%
        [iwpdsigFre,iwpdsigINV] = plot_fft2(Sig,resamplerate, fspan ,mbrokefreq(number,1,i), mbrokefreq(number,2,i),mbrokefreq(number,3,i), mbrokefreq(number,4,i),... 
        mbrokefreq(number,5,i), mbrokefreq(number,6,i), mbrokefreq(number,7,i), mbrokefreq(number,8,i), RMSP,date,motorname,signal_type{signal_type_select},harmonics,i,1,j,fband,plotpiciRE,plotime,mechfalt_num,stringb); 
%     
%        harmonicsV(i,j) = sum(iwpdsigFre(3:end,2));      
%          
%        HighFre(i,j) = sum(iwpdsigFre(harmonics+1:harmonics+1+3,2));
%          
%        HighFreSum(i,j) = sum(iwpdsigINV(Highband(number,:),1));
    
%%  Parks circle
        V = [V1,V2,V3];
        C = [I1,I2,I3];
    
        for o =1:3
            V_LPF(:,o)=(filter_2sIIR(V(:,o)',lp_cut_f,samplerate,6,'low'))'; 
            I_LPF(:,o)=(filter_2sIIR(C(:,o)',lp_cut_f,samplerate,6,'low'))';
        end
    
        K = (2/3)*[1 -1/2 -1/2;0 sqrt(3)/2 -sqrt(3)/2;1/2 1/2 1/2];
        Vdq0 = [];
        for p =1:length(t)
            Cab0(:,p) = K*(C(p,:)');%[C0;Ca;Cb]
        end
    
        data_C = [Cab0(1,:)./max([Cab0(1,:),Cab0(2,:)]);Cab0(2,:)./max([Cab0(1,:),Cab0(2,:)])];
        covmat = cov(data_C');
        cov_PV(i,j) = covmat(1,1)-covmat(2,2);
        crr_PV(i,j) = covmat(1,2);
        cov_PV = abs(cov_PV);
        mean_cov_PV = mean(cov_PV,2);
        maxcov_PV = max(cov_PV,[],2);
%     
%% % ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  
%     br_faltL(i,j) = Fall_c(1,2,j)-Fall_c(harmonics+1,2,j);
%     br_faltR(i,j) = Fall_c(1,2,j)-Fall_c(harmonics+2,2,j);
%     Bro_Bar_trend = min(br_faltL+br_faltR,[],2);
% 
%     fmisL(i,j) =  Fall_c(1,2,j)-Fall_c(harmonics+3,2,j);
%     fmisR(i,j) = Fall_c(1,2,j)-Fall_c(harmonics+4,2,j);
%     Mis_trend = min(fmisR+fmisL,[],2);
%     
%     
%     
%     
%     for p=1:1:floor((harmonics-1)/2)
%     foddharmonic(i,j,p) =  Fall_c(1,2,j)-Fall_c(1+2*p,2,j);    
%     end
    
     f_total_oddharmonic = sum(foddharmonic,3);
%% ------------------------------------------------------------------------------------------------------------------------------------------------ 取頻帶 的指標 ---------------------------------------------------------------------------------------------------------------------------------------     
        
        
      fspanvar1(j,i) = var(I_V_fft((floor(mbrokefreq(number,4,6)*10)):floor(mbrokefreq(number,2,6)*10),j));
      fspanvar2(j,i) = var(I_V_fft((floor(mbrokefreq(number,3,6)*10)):floor(mbrokefreq(number,5,6)*10),j));
     %%  軸承 fft 頻帶 
     % fft 主頻 - 軸承損壞頻率 的 variance 值
      fspanvar3(j,i) = var(Fall_c(1,2,j)-I_fft((floor((mbrokefreq(number,6,6)-dis)*10):floor((mbrokefreq(number,6,6)+dis)*10)),j));   % 內還
      fspanvar4(j,i) = var(Fall_c(1,2,j)-I_fft((floor((mbrokefreq(number,7,6)-dis)*10):floor((mbrokefreq(number,7,6)+dis)*10)),j));     % 外還
      fspanvar5(j,i) = var(Fall_c(1,2,j)-I_fft((floor((mbrokefreq(number,8,6)-dis)*10):floor((mbrokefreq(number,8,6)+dis)*10)),j));     % 滾珠
     % fft 主頻 - 軸承損壞頻率 的 mean 值
      fspanmean3(j,i) = mean(Fall_c(1,2,j)-I_fft((floor((mbrokefreq(number,6,6)-dis)*10):floor((mbrokefreq(number,6,6)+dis)*10)),j));   % 內還
      fspanmean4(j,i) = mean(Fall_c(1,2,j)-I_fft((floor((mbrokefreq(number,7,6)-dis)*10):floor((mbrokefreq(number,7,6)+dis)*10)),j));     % 外還
      fspanmean5(j,i) = mean(Fall_c(1,2,j)-I_fft((floor((mbrokefreq(number,8,6)-dis)*10):floor((mbrokefreq(number,8,6)+dis)*10)),j));     % 滾珠
      
      fspanvar_cv3(j,i) = var(I_V_fft((floor((mbrokefreq(number,6,6)-dis)*10):floor((mbrokefreq(number,6,6)+dis)*10)),j));        % 內還
      fspanvar_cv4(j,i) = var(I_V_fft((floor((mbrokefreq(number,7,6)-dis)*10):floor((mbrokefreq(number,7,6)+dis)*10)),j));       % 外還
      fspanvar_cv5(j,i) = var(I_V_fft((floor((mbrokefreq(number,8,6)-dis)*10):floor((mbrokefreq(number,8,6)+dis)*10)),j));      % 滾珠
      
      

      
      %% 轉子
      fspankur1(j,i) = kurtosis(I_fft((floor((mbrokefreq(number,1,6)-dis)*10):floor((mbrokefreq(number,1,6)+dis)*10)),j));      
      fspankur3(j,i) = kurtosis(I_fft((floor((mbrokefreq(number,1,6)*3-dis)*10):floor((mbrokefreq(number,1,6)*3+dis)*10)),j));
      fspankur5(j,i) = kurtosis(I_fft((floor((mbrokefreq(number,1,6)*5-dis)*10):floor((mbrokefreq(number,1,6)*5+dis)*10)),j));
      fspankur = fspankur1+fspankur3+fspankur5;
      %%  每一筆 data 的值
%       for k = 1:1:3
%       mean_wpd_I(k,j,i) = mean(Fault_c_wpd_coefficient(k,:,j)./Fault_v_wpd_coefficient(k,:,j));
%       rms_wpd_I(k,j,i) = rms(Fault_c_wpd_coefficient(k,:,j)./Fault_v_wpd_coefficient(k,:,j));
%       std_wpd_I(k,j,i) = std(Fault_c_wpd_coefficient(k,:,j)./Fault_v_wpd_coefficient(k,:,j));
%       mean_wpd_IV(k,j,i) = mean(Fault_c_wpd_coefficient(k,:,j)./Fault_v_wpd_coefficient(k,:,j));
%       rms_wpd_IV(k,j,i) = rms(Fault_c_wpd_coefficient(k,:,j)./Fault_v_wpd_coefficient(k,:,j));
%       std_wpd_IV(k,j,i) = std(Fault_c_wpd_coefficient(k,:,j)./Fault_v_wpd_coefficient(k,:,j));
%       end
%       fspandiff = 2^level;
%       wpt = wpdec(I1,level,'dmey');
%       [Spec,Time,Freq] = wpspectrum(wpt,samplerate);  
%       A= wenergy(wpt);
%       AA(j,i) = A(39); 
      
    end
%% --------------------------------------------------------------------------------------------------------------------------------------- 計算每次量測資料的平均值----------------------------------------------------------------------------------------------
    current_harmonics(:,i) = sum(10.^(Fall_c(2:harmonics,2,:)/20),3)/amount;
    
    Average_Fall_c(:,:,i) =  sum(Fall_c,3)/amount;
    Average_Fall_c_v(:,:,i) =  sum(Fall_c_v,3)/amount;
    
    Average_I_fft(:,i) = sum(I_fft,2)/amount;
    Average_I_V_fft(:,i) = sum(I_V_fft,2)/amount;
        
    average_thd_dbV = Total_thd_dbV/amount; 
    percent_thdV(i) = 100*(10^(thd_dbV/20));
    
    average_thd_dbI = Total_thd_dbI/amount; 
    percent_thdI(i) = 100*(10^(thd_dbI/20));
    
    percent = percent_thdI./percent_thdV;
     
    Average_I(:,i) = SumI/amount;
    Average_V(:,i) = SumV/amount;
    Average_P(:,i) = sum(Power(:,i))/amount;
    
%% ---------------------------------------------------------------------------------------------------------------------------------------------計算 RMS 之 value --------------------------------------------------------------------------------------------------------------------------------------------------------
    
    Average_rms_V(i) = sum(RMSV(:,i),1)/amount;
    Average_rms_I(i) = sum(RMSI(:,i),1)/amount;
    Average_rms_P(i) = sum(RMSP(:,i),1)/amount;
    
    Average_Fall_c(:,:,i) = sum(Fall_c,3)/amount;
    Average_Fall_c_v(:,:,i) = sum(Fall_c_v,3)/amount;
    
    Average_Fault_c_wpd_coefficient(:,:,i) = sum(Fault_c_wpd_coefficient,3)/amount;
    Average_Fault_v_wpd_coefficient(:,:,i) = sum(Fault_v_wpd_coefficient,3)/amount;
    


clear data; 
end
%% -----------------------------------------------------------------------------------------------------------------------------------------------------------------
 
Hf950 = mean(AA,1);

sumHigh = sum(HighFreSum,2);
sumHAR =  sum(harmonicsV,2);
Movingterm = 2;

for j=0:1:period-Movingterm
    MovingHarmonic(j+1) = mean(sumHAR(1:j+Movingterm));
    MovingHighFresum(j+1) = mean(sumHigh(1:j+Movingterm));
    MovingHf950(j+1) = mean(Hf950(1:j+Movingterm));
end


figure
bar(MovingHf950)
title('Motor 11 High frequency band energy')
ylabel('Energy(A)')
xlabel('Time to failure')
figure
bar(MovingHarmonic)
title('Motor 11 Harmonics energy')
ylabel('Energy')
xlabel('Time to failure')





thd_trend = mean(thd_total,2);

cv_inner_var = sum(fspanvar_cv3,1);
cv_outer_var = sum(fspanvar_cv3,1);
cv_rolling_var = sum(fspanvar_cv3,1);
cv_bearing = max([cv_inner_var;cv_outer_var;cv_rolling_var],[],1);

    
I_Inner = sum(fspanvar3,1);
I_outer = sum(fspanvar4,1);
I_rolling = sum(fspanvar5,1);
IV_Inner = sum(fspanvar_cv3,1);
IV_outer = sum(fspanvar_cv4,1);
IV_rolling = sum(fspanvar_cv5,1);
IV_misl = sum(fspanvar1,1);
IV_misr = sum(fspanvar2,1);
I_BR = sum(fspankur,1) ;

Amount_harmonics = sum(current_harmonics,1);

Falt_c = zeros(12,period);
Falt_c_v = zeros(12,period);

for i=1:1:period
    for j=1:1:harmonics+7-1
        Falt_c(1,i) = Average_Fall_c(1,2,i);
        Falt_c(j+1,i) = Average_Fall_c(1,2,i) - Average_Fall_c(j+1,2,i);
        Falt_c_v(1,i) = Average_Fall_c_v(1,2,i);
        Falt_c_v(j+1,i) = Average_Fall_c_v(j+1,2,i)-Average_Fall_c_v(1,2,i);
    end
end
  
barl = cell(period,1);
for i=1:1:period
    barl{i} = [date{i},'| Power RMS value: ',num2str(RMSP(i))];  
end

c1 =  categorical(date);
    
    
    %% ----------------------------------------------------------------------------------------------------------------------算 WPD 的 variance rms kurtosis------------------------------------------------------------------------------------------------ 
rms_wpd_cdv = zeros(period,number_wpt_fre);
rms_wpd_c = zeros(period,number_wpt_fre);
std_wpd_cdv = zeros(period,number_wpt_fre);
std_wpd_c = zeros(period,number_wpt_fre);
kurtosis_wpd_cdv = zeros(period,number_wpt_fre);
kurtosis_rms_wpd_c = zeros(period,number_wpt_fre);
mean_wpd_cdv = zeros(period,number_wpt_fre);
mean_wpd_c = zeros(period,number_wpt_fre);

%% Statisitcal Calaultation
for i=1:1:number_wpt_fre
   for j=1:1:period
    rms_wpd_cdv(j,i) = rms(Average_Fault_c_wpd_coefficient(i,:,j)./Average_Fault_v_wpd_coefficient(i,:,j));
    rms_wpd_c(j,i) = rms(Average_Fault_c_wpd_coefficient(i,:,j));
    std_wpd_cdv(j,i) = std(Average_Fault_c_wpd_coefficient(i,:,j)./Average_Fault_v_wpd_coefficient(i,:,j));
    std_wpd_c(j,i) = std(Average_Fault_c_wpd_coefficient(i,:,j));
    kurtosis_wpd_cdv(j,i) = kurtosis(Average_Fault_c_wpd_coefficient(i,:,j)./Average_Fault_v_wpd_coefficient(i,:,j));
    kurtosis_rms_wpd_c(j,i) = kurtosis(Average_Fault_c_wpd_coefficient(i,:,j));
    mean_wpd_cdv(j,i) = mean(Average_Fault_c_wpd_coefficient(i,:,j)./Average_Fault_v_wpd_coefficient(i,:,j));
    mean_wpd_c(j,i) = mean(Average_Fault_c_wpd_coefficient(i,:,j));
   end
end

if plotpictrend == 1
    
%%
    figure('Name','RMS value at different dates')
    subplot(3,1,1)
    bar(c1,Average_rms_V)
    set(gca,'xticklabel',date)
    ylabel('Voltage')
    title('Input Voltage RMS value')
    subplot(3,1,2)
    bar(c1,Average_rms_I)
    set(gca,'xticklabel',date)
    title('Output Current RMS value')
    ylabel('Ampere')
    subplot(3,1,3)
    bar(c1,Average_rms_P)
    set(gca,'xticklabel',date)
    title('Power RMS value')
    ylabel('Watt')
    
    
    figure
    plot(mean(harmonicsV,2))
    xlabel('date')
    set(gca,'xticklabel',date,'FontSize',12)
    title('Motor 11 Harmonics without main frequency')


    
    
    
    
    
    
    
%%
    figure('Name',['The magnitude of main frequency of Motor ',motorname,'on different times'])
    bar(reshape(Average_Fall_c(1,2,:),[period,1]))
    hold on
    ylim([0,100])
    ylabel('Current spectrum (db scale)','FontSize',22)
    set(gca,'xticklabel',date,'FontSize',22)
    title(['The magnitude of main frequency ','(',signal_type{signal_type_select},')',' of Motor ',motorname,' on different date'],'FontSize',12)
    set(gcf,'units','points','position',[x0,y0,width,height])
%   主頻的倍頻值
    figure('Name',['The magnitude  main frequency minus mag of 2~5 times frequencies of Motor ',motorname,'on different dates'])
    bar([Falt_c(2,:);Falt_c(3,:);Falt_c(4,:);Falt_c(5,:)])
    hold on
    set(gca,'xticklabel',{'2','3','4','5'},'FontSize',22)
    ylim([0,100])
    ylabel('Current spectrum (db scale)'); xlabel('Multiplier','FontSize',22)
    title(['The magnitude of 2~5 times main frequencies ','(',signal_type{signal_type_select},')',' of Motor ',motorname,' on different date'],'FontSize',12)
    set(gcf,'units','points','position',[x0,y0,width,height])
    l1 = legend(barl{1},barl{2},barl{3},'Location','northeast');
    l1.FontSize = 16;
 %%    主頻 減去 偏心損壞頻率的 峰值
    figure('Name',['The magnitude of mag of main freq minus I characterisitc freq of Motor ',motorname,'on different date'])
       bar([Falt_c(harmonics+3,:);Falt_c(harmonics+4,:)]);
       ylabel('Current spectrum(db scale)','FontSize',22) ; 
       hold on
     set(gca,'xticklabel',{'Misalignment(L)','Misalignment(R)'},'FontSize',22)
      ylim([0 100])
     title(['The magnitude of I/V characterisitc freq minus mag of main freq of Motor ',motorname,' on different date'],'FontSize',12)
     set(gcf,'units','points','position',[x0,y0,width,height])
    l2 = legend(barl,'Location','northwest');
    l2.FontSize = 12;
     
     %%  -----------------------------------------------------------------------------------------------------------------------------電流/電壓-------------------------------------------------------------------------------------------------------
    % 主頻能量減去損壞特徵頻帶能量之峰值
     figure('Name',['The magnitude of broken characteristic frequency ','(',signal_type{signal_type_select},')','  of Motor ',motorname,' on different date'])
        bar([Falt_c(harmonics+5,:);Falt_c(harmonics+6,:);Falt_c(harmonics+7,:)])
      hold on
      ylabel('Multiplier(db scale)','FontSize',28) ; 
       ylim([0 120])
     set(gca,'xticklabel',{'Inner bearing broken','Outer bearing broken','Rolling element borken'},'FontSize',14)
     title(['The magnitude of broken characteristic frequency ','(',signal_type{signal_type_select},')',' of Motor ',motorname,' on different date'],'FontSize',12)
     set(gcf,'units','points','position',[x0,y0,width,height])
    l3 = legend(barl,'Location','northwest');
    l3.FontSize = 10;
      
%% 畫 bearing fault WPD 的 variance rms kurtosis 的趨勢圖 
    figure('Name',['RMS of current WPD coefficient of the characteristic frequency of broken bearing of',motorname,'on different date'])
    bar(sum(rms_wpd_c,2))
    axis 'auto y'
    ylabel('Ampere','FontSize',22)
    ylim([0 60])
    set(gcf,'units','points','position',[x0,y0,width,height])
%     set(gca,'xticklabel',{'Inner bearing broken','Outer bearing broken','Rolling element borken'},'FontSize',22)
    title(['RMS of current WPD coefficient of the characteristic frequency of broken bearing of motor ',motorname,' on different date'],'FontSize',12)
    l4 = legend(barl,'Location','northwest');  
    l4.FontSize = 16;
 
     figure('Name',['RMS of Current WPD coefficient divided by voltage WPD coefficient','(',signal_type{signal_type_select},')',' of motor ',motorname,' on different times'])
     bar(sum(rms_wpd_cdv,2))
     ylabel('Mutiplier','FontSize',22)
     axis 'auto y'
     ylim([0 10])
%      set(gca,'xticklabel',{'Inner bearing broken','Outer bearing broken','Rolling element borken'},'FontSize',22)
     set(gcf,'units','points','position',[x0,y0,width,height])
     title(['RMS of Current WPD coefficient divided by voltage WPD coefficient ',' of motor ',motorname,' on different times'],'FontSize',12)
     l5 = legend(barl,'Location','northwest');    
     l5.FontSize = 16;
         
     
     %%
     figure('Name',[' Mean of Current WPD coefficient divided by voltage WPD coefficient','(',signal_type{signal_type_select},')',' of motor ',motorname,' on different times'])
     bar(sum(mean_wpd_cdv,2))
     ylabel('Mutiplier','FontSize',22)
     axis 'auto y'
     ylim([0 10])
%      set(gca,'xticklabel',{'Inner bearing broken','Outer bearing broken','Rolling element borken'},'FontSize',22)
     set(gcf,'units','points','position',[x0,y0,width,height])
     title(['Mean of Current WPD coefficient divided by voltage WPD coefficient ',' of motor ',motorname,' on different times'],'FontSize',12)
     l5 = legend(barl,'Location','northwest');    
     l5.FontSize = 16;
     
     
      %% Rolling element
         figure('Name',['Mean of current WPD coefficient of the characteristic frequency of broken bearing of',motorname,'on different date'])
    bar(sum(mean_wpd_c,2))
    axis 'auto y'
    ylabel('Ampere','FontSize',22)
    ylim([0 15])
    set(gcf,'units','points','position',[x0,y0,width,height])
%     set(gca,'xticklabel',{'Inner bearing broken','Outer bearing broken','Rolling element borken'},'FontSize',12)
    title(['Mean of current WPD coefficient of the characteristic frequency of broken bearing of motor ',motorname,' on different date'],'FontSize',12)
    l4 = legend(barl,'Location','northwest');  
    l4.FontSize = 16;
    %% Rolling element 
    
    figure('Name',['STD of current FFT of the characteristic frequency of broken bearing of',motorname,'on different date'])
    bar(cv_bearing)
    axis 'auto y'
    ylabel('Ampere','FontSize',22)
    set(gcf,'units','points','position',[x0,y0,width,height])
    set(gca,'xticklabel',date,'FontSize',22)
    title(['STD of current FFT of the characteristic frequency of broken bearing of ',motorname,' on different date'],'FontSize',12)
%     l4 = legend(barl,'Location','northwest');  
%     l4.FontSize = 16; 
    %% THD
    figure('Name',['Toatal Voltage Harmonics Distortion of',motorname,'on different date'])
    bar(thd_trend)
    axis 'auto y'
    ylabel('Ampere','FontSize',22)
    ylim([0 5])
    set(gcf,'units','points','position',[x0,y0,width,height])
    set(gca,'xticklabel',date,'FontSize',12)
    title(['STD of current WPD coefficient of the characteristic frequency of broken bearing of motor ',motorname,' on different date'],'FontSize',12)
end
    
 figure('Name','THD trend')
 bar(mean(thd_total,2))   
  ylabel('THD (%)','FontSize',22)
    ylim([0 50])
set(gca,'xticklabel',date,'FontSize',12)
 title([motorname,' 號機 THD on different date'],'FontSize',12)

unbalanceII = max(unbalancedI,[],1);
unbalanceIII = mean(unbalanceII,3);
unbalanceVV = mean(unbalancedV,3);
unbalanceVVV = max(unbalanceVV,[],1);
figure;bar(unbalanceIII*100);title('Current Imbalance of three phase on motor 6')
 ylabel('percentage(%)');
 ylim([0,2])

 figure;bar(unbalanceVVV*100);title('Voltage Imbalance of three phase on motor 1')
 ylabel('percentage(%)');
 ylim([0,2])
 
 
 

