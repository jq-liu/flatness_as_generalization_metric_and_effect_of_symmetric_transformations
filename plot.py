import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


path = './plot/'

plot_figure = path + 'figure/'
if not os.path.exists(plot_figure):
	os.makedirs(plot_figure)


def plot(X_path, Y_path, X_label, Y_label, Title, name):
	x = np.load(X_path)
	y = np.load(Y_path)


	slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
	plt.plot(x, slope * x + intercept, 'c--', linewidth=0.5)
	r_value = round(r_value, 3)
	plt.annotate(r'$\rho=$' + str(r_value), xy=(np.average(x)/1.5, np.average(y)/0.6), xycoords='data', xytext=(+30, -30),
				 textcoords='offset points', fontsize=10)
	tau = round(kendall_coeﬃcient(x, y), 3)
	plt.annotate(r'$\tau=$' + str(tau), xy=(np.average(x)/1.5, np.average(y) /0.63), xycoords='data', xytext=(+30, -30),
				 textcoords='offset points', fontsize=10)
	x = np.reshape(x, (-1, 20))
	y = np.reshape(y, (-1, 20))
	plt.xlabel(X_label)
	plt.ylabel(Y_label)
	plt.ylim((0.1, 0.4))
	plt.title(Title)

	h = []
	for i in range(len(x)):
		plting = plt.scatter(x[i], y[i])
		h.append(plting, )
	plt.legend(handles=h, labels=['bs=1000,lr=0.02', 'bs=2000,lr=0.04', 'bs=4000,lr=0.08', 'bs=8000,lr=0.16'], loc='lower right')
	# plt.legend(handles=h, labels=['100-100-100-10', '100-100-100-100-10', '100-100-100-100-100-10'], loc='lower right')
	# plt.legend(handles=h, labels=['50-50-50-50-10', '75-75-75-75-10', '100-100-100-100-10'], loc='lower right')
	plt.savefig(plot_figure + name + '.png')
	plt.clf()



if __name__ == "__main__":
	Y = path + 'data/gen_loss.npy'
	layers = 4
	for i in range(layers):
		print(i)
		X = path + 'data/l_max_' + str(i) + '.npy'
		plot(X, Y, r'Neuron-wise measure $\rho^l$', 'Generalization error', 'Layer ' + str(i+2), 'm_' + str(i+2))
		X = path + 'data/l_sum_' + str(i) + '.npy'
		plot(X, Y, r'Neuron-wise measure $\rho^l_{\sigma}$', 'Generalization error', 'Layer ' + str(i+2), 's_' + str(i+2))

	X = path + 'data/n_max.npy'
	plot(X, Y, r'Neuron-wise measure $\rho^{\max}$', 'Generalization error', '', 'm')
	X = path + 'data/n_sum.npy'
	plot(X, Y, r'Neuron-wise measure $\rho^{\sum}$', 'Generalization error', '', 's')



