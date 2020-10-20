from sklearn.decomposition import PCA

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class pca_reducer:
    def __init__(self):
        self.pca = PCA(n_components=6)

    def create_transform(self, X):
        self.transform = self.pca.fit(X)

    def info(self):
        print("explained_variance: {}".format(self.pca.explained_variance_))
        print("explained_variance_ratio: {}".format(self.pca.explained_variance_ratio_))
        print("mean: {}".format(self.pca.mean_))
        print("noise_variance: {}".format(self.pca.noise_variance_))

    def transform_enc(self, X):
        return self.transform.transform(X)
    
    def inverse_pca(self, X):
        return self.transform.inverse_transform(X)

    def save_transform(self,path):
        with open(path, 'wb') as f:
            pickle.dump(self.transform, f, pickle.HIGHEST_PROTOCOL)

    def load_embedder(self,path):
        with open(path, 'rb') as f:
            self.transform = pickle.load(f)

    def plot_cum_var(self):
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()

    def plot_main_components(self, set0, set1):
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.scatter(
            set0[:,0], set0[:,1], c='r', cmap="Spectral", s=6.0, label="val"
        )
        plt.scatter(
            set1[:,0], set1[:,1], c='b', cmap="Spectral", s=4.0, label="train"
        )
        ax.legend()
        ax.set_xlabel("PC0: {}%".format(int(self.pca.explained_variance_ratio_[0]*100)))
        ax.set_ylabel("PC1: {}%".format(int(self.pca.explained_variance_ratio_[1]*100)))
        plt.title("PC0 vs PC1, latent space reduced by PCA", fontsize=18)
        plt.show()

    def plot_all_components(self, set0, set1):
        _, num_dim = set0.shape
        fig, axs = plt.subplots(figsize=(18, 3), ncols=num_dim)
        for id in range(num_dim):
            sns.distplot(set0[:,id], hist=False, rug=True, color="r", ax=axs[id])
            sns.distplot(set1[:,id], hist=False, rug=True, color="b", ax=axs[id])
            axs[id].yaxis.set_visible(False)
        plt.show()

if __name__ == '__main__':
    data_dir = '/home/markpp/github/latent_3d_points/scripts/output/'
    val_enc = np.load(data_dir+'val_latent.npy')[:]
    train_enc = np.load(data_dir+'train_latent.npy')[:]

    # pca dim reduction
    pca_red = pca_reducer()
    pca_red.create_transform(train_enc)
    pca_red.save_transform('pca_transform.pkl')
    train_pca = pca_red.transform_enc(train_enc)
    np.save("train_pca.npy", train_pca)
    val_pca = pca_red.transform_enc(val_enc)
    np.save("val_pca.npy", val_pca)
    #pca_red.plot_cum_var()
    pca_red.plot_main_components(val_pca[:,:],train_pca[:,:])
    pca_red.plot_all_components(val_pca[:,:],train_pca[:,:])
    pca_red.info()