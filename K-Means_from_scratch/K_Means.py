import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import threading

def on_start(file):
    plt.scatter(file['Card Debt'], file['Years Employed'])
    plt.show()

def centers_init(n_clusters):
    clusters = {}; centers = []
    for i in range(n_clusters):
        ind = str(i + 1)
        clusters[ind] = []
        centers.append(features[np.random.randint(0, 569)])
    return centers, clusters

def clear_clusters(clust):
    for i in clust:
        clust[i] = []
    return clust

def K_means(n_clusters, features):    
    centers, clusters = centers_init(n_clusters)
    m = [0, 0, 0]
    while(m[:] != centers[:]):
        clusters = clear_clusters(clusters)
        for i in range(len(features)):
            distances = []; count = 0
            for j in centers:
                count += 1
                distances.append([np.linalg.norm(np.array(features[i]) - np.array(j)), count])
            clusters[str(sorted(distances)[0][1])].append(features[i])
        count_centres = []
        m = centers[:]
        for i in clusters:
            j = np.array(clusters[i])
            count_centres.append(list(np.sum(j, 0)/j.shape[0]))
            centers = count_centres
    return clusters


if __name__ == "__main__":
    file = pd.read_csv("/home/thedarkcoder/Desktop/ML/Coursera/datasets/Cust_Segmentation.csv")
    file.fillna(0)
    thr1 = threading.Thread(target= on_start(file), args= (1), daemon= True)

    features = file[['Card Debt', 'Years Employed']]
    features = features.values.tolist()
    n_clusters = int(input())
    thr1.start()

    clusters = K_means(n_clusters, features)
    color_dict = {'1': "blue", '2': "orange", '3': "green", '4': "yellow", '5' : "purple", '6' : "red", '7' : "black"}

    thr1.join()
    for i in clusters:
        j = np.array(clusters[i])
        plt.scatter(j[:,0], j[:,1], c = color_dict[i])
    plt.show()

