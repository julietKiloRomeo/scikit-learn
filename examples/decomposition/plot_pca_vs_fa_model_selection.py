"""
===============================================================
Model selection with Probabilistic PCA and Factor Analysis (FA)
===============================================================

Probabilistic PCA and Factor Analysis are probabilistic models.
The consequence is that the likelihood of new data can be used
for model selection and covariance estimation.
Here we compare PCA and FA with cross-validation on low rank data corrupted
with homoscedastic noise (noise variance
is the same for each feature) or heteroscedastic noise (noise variance
is the different for each feature). In a second step we compare the model
likelihood to the likelihoods obtained from shrinkage covariance estimators.

One can observe that with homoscedastic noise both FA and PCA succeed
in recovering the size of the low rank subspace. The likelihood with PCA
is higher than FA in this case. However PCA fails and overestimates
the rank when heteroscedastic noise is present. Under appropriate
circumstances the low rank models are more likely than shrinkage models.

The automatic estimation from
Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
by Thomas P. Minka is also compared.

"""

# Authors: Alexandre Gramfort
#          Denis A. Engemann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

print(__doc__)

# #############################################################################
# Create the data

n_samples, n_features, rank = 500, 25, 10
sigma = 1.
rng = np.random.RandomState(42)
U, _, _ = linalg.svd(rng.randn(n_features, n_features))
X = np.dot(rng.randn(n_samples, rank), U[:, :rank].T)

# Adding homoscedastic noise
X_homo = X + sigma * rng.randn(n_samples, n_features)

# Adding heteroscedastic noise
sigmas = sigma * rng.rand(n_features) + sigma / 2.
X_hetero = X + rng.randn(n_samples, n_features) * sigmas

# #############################################################################
# Fit the models

n_components = np.arange(0, n_features, 5)  # options for n_components

def compute_scores(X):
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=3)))
        fa_scores.append(np.mean(cross_val_score(fa, X, cv=3)))

    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages}, cv=3)
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X, cv=3))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X, cv=3))


fig, (left, mid, right) = plt.subplots(1, 3, figsize=(9, 3))
for X, title, ax in [(X_homo, 'Homoscedastic Noise', left),
                     (X_hetero, 'Heteroscedastic Noise', right)]:
    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = PCA(svd_solver='full', n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("- "*3 + title + " -"*3)
    print("      method         best n " % n_components_pca)
    print("- "*17)
    print("      PCA CV          %d" % n_components_pca)
    print("      PCA MLE         %d" % n_components_pca_mle)
    print("      FA CV           %d" % n_components_fa)
    print("      truth           %d\n" % rank)

    ax.plot(n_components, pca_scores, 'b', label='PCA scores')
    ax.plot(n_components, fa_scores, 'r', label='FA scores')
    ax.axvline(rank, color='g', label='TRUTH', linestyle='-', lw=3, alpha=0.5)
    ax.axvline(n_components_pca, color='b',
               label='PCA CV: %d' % n_components_pca, linestyle='--')
    ax.axvline(n_components_fa, color='r',
               label='FA CV',
               linestyle='--')
    ax.axvline(n_components_pca_mle, color='k',
               label='PCA MLE', linestyle='--')

    # compare with other covariance estimators
    ax.axhline(shrunk_cov_score(X), color='violet',
                label='Shrunk Covariance MLE', linestyle='-.')
    ax.axhline(lw_score(X), color='orange',
                label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    ax.set_xlabel('nb of components')
    ax.set_ylabel('CV scores')
    ax.set_title(title)

right.yaxis.tick_right()
right.yaxis.set_label_position("right")

left.legend(
    loc='upper right',
    bbox_to_anchor=(1.8, 0, 0.5, 1),
    frameon=False)
mid.set_axis_off()

plt.show()
