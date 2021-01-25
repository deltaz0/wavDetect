
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import os

system = 57779
newpath = "./" + str(system) + "figs"
if not os.path.exists(newpath):
    os.makedirs(newpath)
resultsFileName = 'resultscurrent.csv'
results = np.genfromtxt(resultsFileName)
results = results[-32:,:]
print(results.shape)

plt.plot(np.divide(results[:,4] - results[:,5],results[:,8]),label='$c_{H_i}$')
plt.plot(np.divide(results[:,1] - results[:,0],results[:,6]),label='$c_{K_i}$')
plt.plot(np.divide(results[:,3] - results[:,2],results[:,7]),label='$c_{U_i}$')
plt.legend()
plt.title("Avg Means")
plt.ylim(-0.05,0.2)
plt.legend()
plt.title("Scaled Mean Comparisons (unaltered case)")
plt.ylabel('Mean Difference Scaled by Mean Avg')
plt.xlabel('Iteration')
plt.savefig(str(system) + 'figs/method2UnAlt.png')
plt.show()
plt.clf()

plt.plot(np.divide(results[:, 40] - results[:, 41], results[:, 45]), label='$c_{u_i}$ $/$ $a_{u_i}$')
plt.plot(np.divide(results[:, 39] - results[:, 41], results[:, 46]), label='$c_{k_i}$ $/$ $a_{k_i}$')
plt.ylim(-0.05,0.2)
plt.legend()
plt.title("Scaled Mean Comparisons (unaltered case)")
plt.ylabel('Mean Difference Scaled by Mean Avg')
plt.xlabel('Iteration')
plt.savefig(str(system) + 'figs/method1Unalt.png')
plt.show()
plt.clf()

system = 57779
newpath = "./" + str(system) + "figs"
if not os.path.exists(newpath):
    os.makedirs(newpath)
resultsFileName = 'results/' + str(system) + 'system/resultsa13.csv'
results = np.genfromtxt(resultsFileName)
results = results[-32:,:]
print(results.shape)

plt.plot(np.divide(results[:,4] - results[:,5],results[:,8]),label='$c_{H_i}$')
plt.plot(np.divide(results[:,1] - results[:,0],results[:,6]),label='$c_{K_i}$')
plt.plot(np.divide(results[:,3] - results[:,2],results[:,7]),label='$c_{U_i}$')
plt.legend()
plt.title("Avg Means")
plt.ylim(-0.05,0.2)
plt.legend()
plt.title("Scaled Mean Comparisons (altered case)")
plt.ylabel('Mean Difference Scaled by Mean Avg')
plt.xlabel('Iteration')
plt.savefig(str(system) + 'figs/method2Alt.png')
plt.show()
plt.clf()

plt.plot(np.divide(results[:, 40] - results[:, 41], results[:, 45]), label='$c_{u_i}$ $/$ $a_{u_i}$')
plt.plot(np.divide(results[:, 39] - results[:, 41], results[:, 46]), label='$c_{k_i}$ $/$ $a_{k_i}$')
plt.ylim(-0.05,0.2)
plt.legend()
plt.title("Scaled Mean Comparisons (altered case)")
plt.ylabel('Mean Difference Scaled by Mean Avg')
plt.xlabel('Iteration')
plt.savefig(str(system) + 'figs/method1Alt.png')
plt.show()