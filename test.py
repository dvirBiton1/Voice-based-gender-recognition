from sklearn import mixture

females_gmm = mixture.GaussianMixture(n_components=16, means_init=(200,200), covariance_type='diag', n_init=3)
males_gmm = mixture.GaussianMixture(n_components=16, means_init=200, covariance_type='diag', n_init=3)
females_gmm.means_= 16,200
print(females_gmm.means_)