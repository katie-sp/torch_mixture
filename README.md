# torch-mixture package

Implementation of EM algorithm for fitting mixture models using pytorch.
Fitting is fast due to the following factors:
* At initialization, each mixture distribution's parameters are initialized to match the moments of the data rather than fitting MLE
* At initialization, if clustering is chosen to separate data into subsets for each cluster, clustering uses the k-means minibatch algorithm that can be run on multiple cores
* EM optimization is implemented in pytorch using torch's Adam optimizer to take one step each iteration rather than carry out full M-step optimization

### Installation:
pip install --index-url https://test.pypi.org/simple/ --no-deps torch-mixture-katies
