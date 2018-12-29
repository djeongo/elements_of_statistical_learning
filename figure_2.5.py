import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sb
import pandas as pd
import itertools
import scipy as sp

N_mu = 10
cov = np.array([[1.0, 0], [0, 1.0]])

def gen_data(means, N):
  data = []
  for i in range(N):
    # pick a mean
    mu = means[np.random.randint(0, N_mu)]
    val = np.random.multivariate_normal(mu, cov/5, size=1)
    data.append([val[0][0], val[0][1]])
  return np.array(data)


N = 100
# class BLUE
blue_mu = [1, 0]
blue_m = np.random.multivariate_normal(blue_mu, cov, size=N_mu)
blue = gen_data(blue_m, N)
# class ORANGE
orange_mu = [0, 1]
orange_m = np.random.multivariate_normal(orange_mu, cov, size=N_mu)
orange = gen_data(orange_m, N)
# classification
classification = []
for x in np.arange(-5, 5, 0.1):
  for y in np.arange(-5 ,5, 0.1):
    p_blue = max([sp.stats.multivariate_normal.pdf([x,y], mean=blue_mu, cov=cov/5) for blue_mu in blue_m])
    p_orange = max([sp.stats.multivariate_normal.pdf([x,y], mean=orange_mu, cov=cov/5) for orange_mu in orange_m])
    result = "orange"
    if p_blue > p_orange:
      result = "blue"
    classification.append({"x":x, "y":y, "class":result, "size":0})
        

df_blue = pd.DataFrame({"x":blue[:,0],
                        "y":blue[:,1],
                        "class":"blue",
                        "size":2})
df_orange = pd.DataFrame({"x":orange[:,0],
                          "y":orange[:,1],
                          "class":"orange",
                          "size":2})
df_class = pd.DataFrame(classification)

df = pd.concat([df_blue, df_orange, df_class])

sb.scatterplot(x="x", y="y", data=df, hue="class", size="size")
plt.show()
