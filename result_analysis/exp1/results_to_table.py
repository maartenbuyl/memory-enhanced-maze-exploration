import numpy as np
import os

results_files = [
    "post_results_lin.txt",
    "post_results_lstm.txt",
    "post_results_mem.txt"
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

for i in range(datas.shape[0]):
    for j in range(datas.shape[1]):
        temp = datas[i, j, 8]
        datas[i, j, 8] = datas[i, j, 9]
        datas[i, j, 9] = temp

output_prefixes = ["success rate", "fail rate", "sloth rate", "recall50", "recall90", "wall collisions",
                   "inefficient turns", "inefficient revisits", "closure inefficiency", "distance inefficiency"]

output = ""
for i in range(len(output_prefixes)):
    output += output_prefixes[i]

    for k in range(2):
        for j in range(len(results_files)):
            output += " & " + str(datas[k, j, i])

    output += "\\\\" + "\n" + "\\hline" + "\n"

print(output)
