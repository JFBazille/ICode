% %load(/volatile/hubert/datas/simulations/simulationsfGn2.mat)
% dfa = zeros(9,1000);
% whittle = zeros(9,1000);
% for i = 1:9
%     for j = 1:1000
%         d = reshape(simulations(i,j,:),1,4096);
%         dfa(i,j) = HDFAEstim(d,1,2,8,0);
%         whittle(i,j) = whittlenew(d);
%     end
% end
% 
% save('/volatile/hubert/datas/simulations/matlab_estimations_4096.mat','dfa','whittle')

for i = 1:9
    for j = 1:1000
        d = reshape(simulations(i,j,1:514),1,514);
        dfa(i,j) = HDFAEstim(d,1,2,8,0);
        whittle(i,j) = whittlenew(d);
    end
end

save('/volatile/hubert/datas/simulations/matlab_estimations_514.mat','dfa','whittle')