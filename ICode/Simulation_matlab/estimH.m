load('/volatile/hubert/datas/signal.mat')
n=length(signal);


H = zeros(n,1);

for i = 1:n
    H(i,1) = whittlenew(reshape(signal(:,i),1,514));
end

save('/volatile/hubert/datas/mretour.mat','H')