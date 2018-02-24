
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from iris_dataset import dataset

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

#histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()
