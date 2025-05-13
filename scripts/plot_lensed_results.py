# imports
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", help = "input suffix", required = True)

args = parser.parse_args()


# load data
trial_dms = np.load(args.i + "_trial_DMs.npy")
corr_vals = np.load(args.i + "_corr_vals.npy")

fig, _ = plt.subplots(1, 1, figsize = (10,8))
plt.plot(trial_dms, corr_vals, 'k')

plt.xlabel("Trial DM [pc cm^-3]", fontsize = 16)
plt.ylabel("Corr val (arb.)", fontsize = 16)

fig.tight_layout()
fig.subplots_adjust(hspace = 0, wspace = 0)

plt.show()