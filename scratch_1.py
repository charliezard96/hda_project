import numpy as np
import matplotlib.pyplot as plt

def extract_history(stringa):

    f = open(stringa, 'r')
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
    dir = 'history\\CLASSIFIER\\newDROPrate04_A_SCandBN_1024_1024_200epochs(MIGLIORE)'
    mod = 'history_newDROPrate04_classifier_A_SCandBN_1204_1024_epochs200.txt'
    a = extract_history(dir+'\\acc_'+mod)
    b = extract_history(dir+'\\val_acc_'+mod)
    c = extract_history(dir+'\\loss_'+mod)
    d = extract_history(dir+'\\val_loss_'+mod)

    fig, ax = plt.subplots(2, 2, figsize=(24, 16))

    ax[0, 0].plot(a)
    ax[0, 0].grid()
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_title('train_acc AE_BN')
    #ax[0, 0].set_title('loss SC and BN')

    ax[0, 1].plot(b)
    ax[0, 1].grid()
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_title('val_acc AE_BN')
    #ax[0, 1].set_title('val loss SC and BN')

    ax[1, 0].plot(c)
    ax[1, 0].grid()
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_title('train_loss AE_BN')
    #ax[1, 0].set_title('loss BN')

    ax[1, 1].plot(d)
    ax[1, 1].grid()
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_title('val_loss AE_BN')
    #ax[1, 1].set_title('val loss BN')

    plt.show()

    z=1

if __name__ == "__main__":
    main()