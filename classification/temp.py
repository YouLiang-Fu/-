import numpy as np
samples = np.loadtxt("kmeansSamples.txt")

from sklearn.mixture import GaussianMixture
X = samples
import time
import time
t0=time.time()
gm = GaussianMixture(n_components=3, random_state=0).fit(X)
t1=time.time()
tn=t1-t0
print(tn)
y_pred = gm.predict(X)

import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()

from sklearn.metrics import silhouette_score
sc_value=silhouette_score(X,y_pred)
print(sc_value)
print(gm.means_)
