import numpy as np
import pandas as pd 

class LinearModel(object):
  def __init__(self, X=None, y=None):
    self.X = np.array(X)
    self.y = np.array(y)

    self.n, self.p = X.shape

    self.rss = None
    self.rse = None

  def fit(self):
    X = self.X
    y = self.y
   
    ones = np.ones((self.n,1))

    X = np.concatenate((X, ones), axis=1)
    
    self.B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    self.y_hat = X.dot(self.B)

    # compute RSS
    self.rss = self.compute_rss()

    # compute RSE
    self.rse = self.compute_rse()

    # compute TSS
    self.tss = self.compute_tss()

    # compute R^2
    self.r2 = self.compute_rsquared()

    # compute F statistic
    self.fstat = self.compute_fstatistic()

  def compute_rss(self):
    '''Compute Residual Sum of Squares
    
    RSS = e1^2 + e2^2 + e3^2 + ... + e4^2
    '''
    rss = np.sum(np.square(self.y - self.y_hat))
    return rss

  def compute_rse(self):
    '''Compute Residual Standard Error

    RSE = sqrt(1/(n-2) * RSS)
    '''
    rse = np.sqrt(1/(self.n-2) * self.rss)
    return rse

  def compute_tss(self):
    '''Compute Total Sum of Squares
    '''
    tss = np.sum(np.square(self.y - np.mean(self.y)))
    return tss

  def compute_rsquared(self):
    '''Compute R^2

    R_squared = (TSS - RSS) / TSS = 1 - RSS/TSS
    '''
    r2 = 1 - self.rss / self.tss
    return r2
  
  def compute_std_error(self):
    pass

  def compute_fstatistic(self):
    F = ((self.tss - self.rss)/self.p) / (self.rss/(self.n - self.p - 1))
    return F

def main():
  df = pd.read_csv('Advertising.csv', index_col=0)
  N, p = df.shape
  print(df.head())
  # Table 3.2
  X_tv,y = df.iloc[:, [0]], df.iloc[:,3]
  
  # Add ones
  lm = LinearModel(X_tv, y)
  lm.fit()
  print(lm.B)
  print(lm.rss)
  print(lm.rse)
  print(lm.r2)
  print(lm.fstat)

if __name__=="__main__":
  main()
