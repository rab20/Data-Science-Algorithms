# Apply Min-Max Normalization to 'Age' values in the Covid dataset; 
# So age ranges will be in the interval [0,1]

import pandas as pd

df = pd.read_csv("Covid_data.csv")

df.head()

# copy the data
df_min_max_scaled = df.copy()

# apply normalization techniques
#for column in df_min_max_scaled.columns:
# new-x = x - min(x) / max(x) - min(x)
df_min_max_scaled['Age'] = (df_min_max_scaled['Age'] - df_min_max_scaled['Age'].min()) / (df_min_max_scaled['Age'].max() - df_min_max_scaled['Age'].min())	

# view normalized data
lst = []
for val in df_min_max_scaled['Age']:
  lst.append(val) 

formatted_lst = ['%.2f' % elem for elem in lst]

#print(df_min_max_scaled['Age'])
print(formatted_lst)