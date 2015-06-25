N =1000;

l = 4096;
addpath /volatile/hubert/schubert/Biyu_code/
addpath /volatile/hubert/schubert/Biyu_code/WHITTLE/
addpath /volatile/hubert/schubert/matlab/fBM/
%new file opening



%simulation is a big matrix with all our simulation
simulations = zeros(9,N,l);

for j = 1:9
    Htheo = j/10;
    for i = 1:N
        %[B,simulations(j,i,:)] = synthfbmcircul(l,Htheo); pour obtenir fGn
        %[simulations(j,i,:),B] = synthfbmcircul(l,Htheo); pour un fBm
        
    end
end
save('/volatile/hubert/datas/simulations/simulationsfBm.mat','N','l', 'simulations')