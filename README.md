# KRR typical-case rates

Code for the paper : <i>Generalization error rates in kernel regression: The crossover from the noiseless to noisy regime</i> (<a href="https://proceedings.neurips.cc/paper/2021/hash/543bec10c8325987595fcdc492a525f4-Abstract.html">link to paper</a>)

<p align="center"><img src="figures/scaling_laws.jpg" alt="illus" width="600"/></center></p>

## Theoretical characterization 
(Figs. 2, 3, solid lines)
- <tt>replica_curves.ipynb</tt> provides a Jupyter notebook implementing the theoretical characterization of equation (14) for the excess risk $\epsilon_g-\sigma^2$, for Gaussian data.

### Numerical experiments
- <tt>Real_KRR.py</tt> implements kernel ridge regression on a given dataset (to be loaded in a folder Datasets/), with the $\ell_2$ regularization strength being optimized over. For instance, to run kernel ridge regression with additive noise $\sigma=0.5$, RBF kernel with parameter $\gamma=0.7$, run
  ```
  python3 Real_KRR.py --p 0.5 --k rbf --r 0.7 --d MNIST --v 6
  ```
  The <tt>--v</tt> parameter can be looped over, it simply runs through a list of sample sizes $n$.
- <tt>Real_KRR_noreg.py</tt> implements the same routine for $\lambda=0$.
-  <tt>Real_KRR_decay.py</tt> provides the same routine, but for a regularization generically decaying with the number of samples as $\lambda=n^{-\ell}$. For instance, to run kernel ridge regression with additive noise $\sigma=0.5$, RBF kernel with parameter $\gamma=0.7$, and regularization decay $\ell=0.1$, run
  ```
  python3 Real_KRR.py --p 0.5 --k rbf --r 0.7 --d MNIST --v 6 --c 0.1
  ```
  
<b>Versions:</b> These notebooks employ <tt>Python 3.12 </tt>, and <tt>Pytorch 2.5</tt>. The numerical experiment use the <tt>scikit-learn GridSearchCV</tt> routine, which uses sklearn 0.22 onwards.
