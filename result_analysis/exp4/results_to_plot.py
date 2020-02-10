import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict

results_files = [
#    "results_lin.txt",
    "results_lstm1.txt",
    "results_lstm2.txt",
    "results_lstm3.txt",
    "results_lstm4.txt",
    "results_lstm5.txt",
    "results_mem1.txt",
    "results_mem2.txt",
    "results_mem3.txt",
    "results_mem4.txt",
    "results_mem5.txt"
]

epoch_cutoff = 1000
SMOOTH_FAC = 10
possible_labels = ["Non-Rec", "LSTM", "SNM"]
w = 7
h = 3.5

plt.style.use('bmh')


# Tuple: (T correct, T wrong, V correct, V wrong)
def file_into_tuple(file_name):
    prefix = os.path.dirname(__file__) + "/"

    file = open(prefix + file_name)
    lines = file.readlines()

    epoch = 0
    result = [[], [], [], []]

    for line in lines:
        index = 2

        if epoch >= epoch_cutoff:
            break

        for i in range(4):
            end_of_value = line.find(",", index)
            result[i].append(float(line[index:end_of_value]))
            if i == 0 or i == 2:
                index = end_of_value + 2
            else:
                index = end_of_value + 3
        epoch += 1

    result = np.array(result)
    return result


def smoother(a, n=SMOOTH_FAC):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = (ret[n:] - ret[:-n]) / n
    for i in range(0, n):
        ret[i] = ret[i] / (i+1)
    return ret


datas = []
for i in range(len(results_files)):
    results_as_tuple = file_into_tuple(results_files[i])
    datas.append(results_as_tuple)
datas = np.array(datas)

#font = {'family' : 'normal',
#        'weight' : 'normal',
#        'size'   : 12}

#matplotlib.rc('font', **font)

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(w, h)

actual_nb_epochs = datas[0].shape[1]+1

i_to_network_type = []
#i_to_network_type.append(0)  # For non-rec
for i in range(5):
    i_to_network_type.append(1)  # For LSTM
for i in range(5):
    i_to_network_type.append(2)  # For SNM

possible_colors = ["#348ABD", "#A60628", "#7A68A6"]

# Success rates
for i in range(2):
    ax[i].set(xlabel="Epochs", ylabel="success rate", ylim=(-0.05, 1.05))
    epochs = np.arange(1, actual_nb_epochs)

    # For non-rec
    #data_j = 0
    #ax[i].plot(epochs, smoother(datas[data_j][i*2]), label=possible_labels[0], color=possible_colors[0])

    # For LSTM
    mean_values = smoother(datas[0:0+5, i*2, :].mean(0))
    ax[i].plot(epochs, mean_values, label=possible_labels[1], color=possible_colors[1])
    for data_j in range(0, 0+5):
        ax[i].plot(epochs, smoother(datas[data_j][i*2]), label=possible_labels[1], color=possible_colors[1], alpha=0.15, ls='-')

    # For SNM
    mean_values = smoother(datas[5:5+5, i*2, :].mean(0))
    ax[i].plot(epochs, mean_values, label=possible_labels[2], color=possible_colors[2])
    for data_j in range(5, 5+5):
        ax[i].plot(epochs, smoother(datas[data_j][i*2]), label=possible_labels[2], color=possible_colors[2], alpha=0.15, ls='-')

    # Remove duplicate labels
    handles, labels = ax[i].get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
        if label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
    ax[i].legend(newHandles, newLabels, frameon=False, framealpha=1)

ax[0].set_title('Training Set', fontweight='bold', size=16, pad=15)
ax[1].set_title('Test Set', fontweight='bold', size=16, pad=15)

plt.subplots_adjust(wspace=0.5, hspace=0.2)
plt.savefig("plot.pdf", bbox_inches='tight')
plt.show()
