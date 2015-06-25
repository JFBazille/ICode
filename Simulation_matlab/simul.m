N =1000;

l = 514;
addpath /volatile/hubert/schubert/Biyu_code/
addpath /volatile/hubert/schubert/Biyu_code/WHITTLE/
addpath /volatile/hubert/schubert/matlab/fBM/
%new file opening



Htheo = 0.8;
%simulation is a big matrix with all our simulation
simulations = zeros(N,l);
for i = 1:N
    [B,simulations(i,:)] = synthfbmcircul(l,Htheo);   
end

save('/volatile/hubert/datas/simulations/simulationsfGn514.mat','N','l', 'simulations')