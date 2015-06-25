d = scio.load('/volatile/hubert/datas/mretour.mat')
Hurst = d['H']

img = masker2.inverse_transform(Hurst)

plt.figure(1)
plot_stat_map(img8)
plt.title('Husrt exponent calculated with matlba function HDFAEstim')



plt.show()