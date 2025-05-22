import matplotlib.pyplot as plt
import pandas as pd


def plot_results(results):

    # plot the results
    fig, ax = plt.subplots()
    mAP = results['map']
    ax.plot(mAP, label='mAP', color='blue')
    ax.set_title('mAP vs Epochs')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    results = pd.read_csv("results.csv")
    plot_results(results)