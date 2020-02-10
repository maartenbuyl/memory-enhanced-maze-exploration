import matplotlib.pyplot as plt
import numpy as np
import os

results_files = [
    "results_lin.txt",
    "results_lstm.txt",
    "results_mem.txt"
]

epoch_cutoff = 1000
SMOOTH_FAC = 10
labels = ["Non-Rec", "LSTM", "SNM"]
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
    for j in range(len(results_as_tuple)):
        results_as_tuple[j] = smoother(results_as_tuple[j])
    datas.append(results_as_tuple)

#font = {'family' : 'normal',
#        'weight' : 'normal',
#        'size'   : 12}

#matplotlib.rc('font', **font)

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(w, h)

actual_nb_epochs = datas[0].shape[1]+1

# Success rates
for i in range(2):
    ax[i].set(xlabel="Epochs", ylabel="success rate", ylim=(-0.05, 1.05))

    for j in range(len(datas)):
        epochs = np.arange(1, actual_nb_epochs)
        ax[i].plot(epochs, datas[j][i*2], label=labels[j])
    ax[i].legend(frameon=False, framealpha=1)


ax[0].set_title('Training Set', fontweight='bold', size=16, pad=15)
ax[1].set_title('Test Set', fontweight='bold', size=16, pad=15)


plt.subplots_adjust(wspace=0.5, hspace=0.2)
plt.savefig("plot.pdf", bbox_inches='tight')
plt.show()

