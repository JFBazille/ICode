N = 4096;
addpath /volatile/hubert/schubert/Biyu_code/
addpath /volatile/hubert/schubert/Biyu_code/WHITTLE/
addpath /volatile/hubert/schubert/matlab/fBM/
%new file opening
g = fopen('/volatile/hubert/datas/simulations/estim_results','w');
fprintf(g,'Htheo \t HWhfGn \t HDFAfGn \t HWhfBm \t HDFAfBm \t HWhMRW \t HDFAMRW \n');

for i = 1:9
    Htheo = i/10;
    %the theorical hurst exponent is writen in file g
    fprintf(g, '%1.2d \t', Htheo);
    %new simulation of fGn and fBm
    [B,data] = synthfbmcircul(N,Htheo);
        %the simulation of fGn is writen in a file
    f = fopen(strcat('/volatile/hubert/datas/simulations/fGn_H',num2str(i)),'w');
    fprintf(f,'%d \n', data);
    fclose(f);
        %we compute the estimate Hurst exponent with the whittlenew method and
        %write it in the g file
    Hest = whittlenew(data);
    fprintf(g, '%1.2d \t', Hest);
        %same with DFA method
    Hest = HDFAEstim(data, 1,2,8,0);
    fprintf(g, '%1.2d \t', Hest);
        %the simulation of fBm is writen in a file etc
    f = fopen(strcat('/volatile/hubert/datas/simulations/fBm_H',num2str(i)),'w');
    fprintf(f,'%4d \n', B);
    fclose(f);
    
    Hest = whittlenew(B);
    fprintf(g, '%1.2d \t', Hest);
    CS =1 ; j1 = 2 ; j2 =8;
    Hest = HDFAEstim(B, CS,j1,j2,0);
    fprintf(g, '%1.2d \t', Hest);
    %same for a MRW
    lambda = sqrt(0.05) ; q = [1 2 3 4 5] ;
    data = mrw(Htheo, lambda,N,N,q);
    f=fopen(strcat('/volatile/hubert/datas/simulations/MRW_H',num2str(i)),'w');
    fprintf(f,'%d \n', data);
    fclose(f);
    
    
    Hest = whittlenew(data);
    fprintf(g, '%1.2d \t', Hest);
    
    Hest = HDFAEstim(data, 1,2,8,0);
    fprintf(g, '%1.2d \t', Hest);
    
    fprintf(g,'\n');
end
fclose(g);