
% load('/volatile/hubert/datas/simulations/simulationsfGn2.mat')
% simul = reshape(simulations(8,1,:),1,4096);
% nbsamples = 4096 ; 
% Htheo = 0.8 ; 
% fignumber = 0 ; 
% Nwt = 1 ; 
% fignumber = 0 ; 
% j1 = 2 ; j2 = 8 ; 
% norm = 1;
addpath /volatile/hubert/schubert/
addpath /volatile/hubert/schubert/
nbsamples = length(simul) ; 
%%HUBERT MODIF : the two following section 'Spectrum' had been commented
%Spectrum 
% [beta,F,scales] = Hspectrum(data,nbsamples,j1,j2,10*fignumber) ; 
% Hest.spec_fft = (1+beta)/2 ;
% Hestv(1)=Hest.spec_fft;
% 
% Spectrum 
% windowsize = fix(nbsamples/4) ; 
% [beta,F,scales] = Hspectrum(data,windowsize,j1,j2,20*fignumber) ; 
% Hest.spec_welch = (1+beta)/2 ; 
% Hestv(2) = Hest.spec_welch;


% %% HUBERT MODIF : those lines had been comment to permit computation
Nwt = 2 ; 
[slope,Vzeta,Q,Yjq,VarlogmuqjS,nj,aest,logmuqj,slop] = MDestimate3(cumsum(simul),Nwt,norm,0,2,j1,j2,10,0);
Hest.wav2_inc = slope/2 ; 
Hestv(5) = Hest.wav2_inc;

[slope,Vzeta,Q,Yjq,VarlogmuqjS,nj,aest,logmuqj,slop] = MDestimate3(simul,Nwt,norm,0,2,j1,j2,20,0);
Hest.wav2 = slope/2 +1;
Hestv(6) = Hest.wav2;