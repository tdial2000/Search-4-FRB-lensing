# imports
import matplotlib.pyplot as plt
import numpy as np

# equations to use
def dm_smear(nFFTarr, f):
   return 0.00000297619 * nFFTarr * 1e-3 / (4.14938e3 * (1/(f - 336/nFFTarr)**2 - 1/f**2))

# print(dm_smear(336, 663.5))
# print(dm_smear(336, 695.5))
# print(dm_smear(336, 751.5))
# print(dm_smear(336, 1103.5))
# print(dm_smear(336, 1463.5))


# nFFTarr = [np.arange(1, 32), np.arange(32, 168), np.arange(168, 1024), np.arange(1024, 6720)]
nFFTarr = np.arange(1, 672)
fig = plt.figure(figsize = (14,10))
# ax = [fig.add_axes([0.08, 0.78, 0.8, 0.21]), fig.add_axes([0.08, 0.54, 0.8, 0.21]), 
#       fig.add_axes([0.08, 0.30, 0.8, 0.21]), fig.add_axes([0.08, 0.06, 0.8, 0.21])]
ax1 = fig.add_axes([0.08, 0.06, 0.8, 0.90])
f = np.arange(664, 1800)
colors = plt.cm.viridis(np.linspace(0, 1.0, f.size))[::-1]

# for i in range(4):
dat = np.zeros((f.size, nFFTarr.size))
for j, fi in enumerate(f):
    dat[j] = dm_smear(nFFTarr, fi)

for j in range(f.size - 1):
    plt.fill_between(nFFTarr, dat[j], dat[j+1], color = colors[j])

# label axes
plt.ylabel("DM [pc $cm^{-3}$]", fontsize = 14)

plt.xlabel("# channels", fontsize = 14)
ax1.set_yscale('log')
    

ax = fig.add_axes([0.90, 0.06, 0.04, 0.93])

ax.imshow(f.reshape(f.size, 1), aspect = 'auto', extent = [0, 1, f[0], f[-1]])
ax.yaxis.tick_right()
ax.set_ylabel("Frequency [MHz]", fontsize = 14)
ax.yaxis.set_label_position("right")
ax.get_xaxis().set_visible(False)

ylim = ax1.get_ylim()
xlim = ax1.get_xlim()
# plot known channelisations and freq lines

ax1.plot([336]*2, ylim, 'k--', label = "nchan = 336")

ax1.plot(nFFTarr, dm_smear(nFFTarr, 919.5 - 168), linestyle = '--', color = "darkviolet", label = "919.5 MHz (cfreq)")
ax1.plot(nFFTarr, dm_smear(nFFTarr, 1271.5 - 168), linestyle = "--", color = "greenyellow", label = "1271.5 MHz (cfreq)")
ax1.plot(nFFTarr, dm_smear(nFFTarr, 1631.5 - 168), linestyle = "--", color = "crimson", label = "1631.5 MHz (cfreq)")

ax1.set_ylim(ylim)
ax1.set_xlim(xlim)
ax1.legend()


fig.tight_layout()
plt.show()

