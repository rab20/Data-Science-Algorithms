# Compute the Pearson Correlation 
import pandas as pd
from scipy.stats import pearsonr
 
# Convert dataframe into series
list1 = [40,21,25,31,38,47]
list2 = [78,70,60,55,80,66]
 
# Apply the pearsonr()
corr, _ = pearsonr(list1, list2)
print('Pearson correlation: %.3f' % corr)

# Pearson correlation: 0.347 (Moderate Positive correlation)
# Interpretaton:
# As the Age increases, Weight increases 


