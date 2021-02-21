import matplotlib.pyplot as plt
import numpy as np
from os import listdir

def extract_history(string):

    f = open(string, 'r')
    temp = []
    for line in f:
        line = line[1:-1]
        temp = line.split(',')
    f.close()

    content = []
    for elem in temp:
        content.append(float(elem))

    return content

def main():
    data_dir = "history\\GRIDSEARCH"
    encoder_dir ="wSCandBatch"
    all_metrics = np.array(listdir(data_dir + "\\" + encoder_dir))


    fig, ax = plt.subplots(2, 2, figsize=(24, 16))
    i = 0
    j = 0
    for dir in all_metrics:

        all_file = np.array(listdir(data_dir + "\\" + encoder_dir + '\\' + dir))
        for f in all_file:
            content = extract_history(data_dir + "\\" + encoder_dir + '\\' + dir + '\\' + f)[1:]
            ax[i, j].plot(content, label=f)
            ax[i, j].grid()
            ax[i, j].set_xlabel('Epoch')
            ax[i, j].set_ylabel(dir)
            ax[i, j].set_title(encoder_dir+'\\'+dir)

        i = i + 1
        if i == 2:
            i = 0
            j = j + 1

    fig.canvas.set_window_title(encoder_dir)
    ax[0, 0].legend()
    plt.show()

    z = 1



if __name__ == "__main__":
    main()