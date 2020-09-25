from sklearn.linear_model import LassoCV
import pandas as pd

# reading the preproc data
x_train = pd.read_csv('x_train.csv')
x_test = pd.read_csv('x_test.csv')  
y_train = pd.read_csv('y_train.csv') 
y_test = pd.read_csv('y_test.csv')

# initialising and fitting the model
model = LassoCV()
model.fit(x_train, y_train)


# Use the forest's predict method on the test data
predictions = model.predict(x_test)

# Calculate the absolute errors
mae = mean_squared_error(y_test, predictions)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', mae)