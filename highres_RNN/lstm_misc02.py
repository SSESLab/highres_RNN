import pandas as pd
import numpy as np
import DataFunctions
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


###################################
c_vector = np.load('Ht2_HVAC1.npy')

c_p2 = c_vector.copy()


print c_p2

pca = PCA(n_components=2)
pca.fit(c_p2)


print pca.explained_variance_
print pca.explained_variance_ratio_
print pca.noise_variance_


#c_p = DataFunctions.normalize_2D(c_vector)

c_p2 = pca.transform(c_p2)
c_min = np.amin(c_p2, axis=-0)
c_max = np.amax(c_p2, axis= 0)

for j in range(0, c_p2.shape[1]):
    c_p2[:, j] = (c_p2[:, j] - c_min[j])/(c_max[j] - c_min[j])



#####clustering
kmeans = KMeans(n_clusters=5).fit(c_p2)


label = kmeans.labels_
inertia_val = kmeans.inertia_

plt.scatter(c_p2[:, 0], c_p2[:, 1])
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(c_p2[:, 0], c_p2[:, 1], c=label, s=50)
#plt.xlim()
ax.set_xlabel('c1')
ax.set_ylabel('c2')
plt.colorbar(scatter)
plt.savefig('c_CRAC1.eps')
plt.show()



print inertia_val




#####################
fig = plt.figure()
t_array = np.arange(0, len(c_p2))
plt.plot(t_array, c_p2[:, 0], 'r-', label='Strategy I', linewidth=1.00)
plt.show()