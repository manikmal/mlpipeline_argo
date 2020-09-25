import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# reading the preproc data
x_train = pd.read_csv('x_train.csv')
x_test = pd.read_csv('x_test.csv')  
y_train = pd.read_csv('y_train.csv') 
y_test = pd.read_csv('y_test.csv')

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data
rf.fit(x_train, y_train)

# Use the forest's predict method on the test data
predictions = rf.predict(x_test)

# Calculate the absolute errors
mae = mean_squared_error(y_test, predictions)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', mae)
