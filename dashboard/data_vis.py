import os, matplotlib.pyplot as plt

def plot_distribution(path):
    classes=os.listdir(path)
    counts=[len(os.listdir(f"{path}/{c}")) for c in classes]
    plt.bar(classes,counts)
    return plt