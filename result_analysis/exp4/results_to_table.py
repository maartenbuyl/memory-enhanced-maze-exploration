import numpy as np
import os

results_files = [
    "post_results_lstm1.txt",
    "post_results_lstm2.txt",
    "post_results_lstm3.txt",
    "post_results_lstm4.txt",
    "post_results_lstm5.txt",
    "post_results_mem1.txt",
    "post_results_mem2.txt",
    "post_results_mem3.txt",
    "post_results_mem4.txt",
    "post_results_mem5.txt",
]


# Output: (training set results, test set results)
def file_into_tuple(file_name):
    prefix = os.path.dirname(__file__) + "/"

    file = open(prefix + file_name)
    lines = file.readlines()

    result = [[], []]

    result_index = 0
    for line in lines:
        if line.startswith("Maze"):
            break
        if line.startswith("Summary of validation set"):
            result_index = 1
        if line.startswith("Summary"):
            continue
        if line.startswith("---"):
            continue

        start_of_value = line.find(":") + 2
        number = float(line[start_of_value:])

        assert number < 1e5

        string = np.format_float_positional(number, precision=5, unique=False, fractional=True)

        result[result_index].append(string)

    result = np.array(result)
    return result

datas = [[], []]
for i in range(len(results_files)):
    results_as_tuple = file_into_tuple(results_files[i])
    datas[0].append(results_as_tuple[0])
    datas[1].append(results_as_tuple[1])
datas = np.array(datas)

output_prefix = "run "

output = ""
for i in range(5):
    output += output_prefix + str(i+1)

    for k in range(2):
        for j in range(2):
            output += " & " + str(datas[k, i + j*5, 0])

    output += "\\\\" + "\n" + "\\hline" + "\n"

lstm_vals = np.array(datas[:, 0:0+5, 0], dtype=np.float)
mean_lstm = lstm_vals.mean(axis=1)
mean_lstm_train = np.format_float_positional(mean_lstm[0], precision=5, unique=False, fractional=True)
mean_lstm_test = np.format_float_positional(mean_lstm[1], precision=5, unique=False, fractional=True)

snm_vals = np.array(datas[:, 5:5+5, 0], dtype=np.float)
mean_snm = snm_vals.mean(axis=1)
mean_snm_train = np.format_float_positional(mean_snm[0], precision=5, unique=False, fractional=True)
mean_snm_test = np.format_float_positional(mean_snm[1], precision=5, unique=False, fractional=True)

output += "mean " + " & " + mean_lstm_train + " & " + mean_snm_train + " & " + mean_lstm_test + " & " + mean_snm_test
output += "\\\\" + "\n" + "\\hline" + "\n"

print(output)


from scipy.stats import ttest_ind

# Compute over test set
print(ttest_ind(lstm_vals[1], snm_vals[1], equal_var=False))
