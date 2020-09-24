import pandas as pd
from sklearn.model_selection import train_test_split


#any data 
df = pd.read_csv('~/Desktop/sales.csv')
#for this small example, I will just remove a column as data preproc
df.drop('size', inplace=True)

x = df.drop('sales', axis=1)
y = df['sales']

x_train, x_test, y_train, y_test = train_test_split(df, test_size=0.3)

df.to_csv('x_train.csv')
df.to_csv('x_test.csv')
df.to_csv('y_train.csv')
df.to_csv('y_test.csv')

