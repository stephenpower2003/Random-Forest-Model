# Import statements
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Path of the file for pandas to read
file_path = "../input/home-data-for-ml-course/train.csv"
data = pd.read_csv(file_path)

# Create target
t = data.SalePrice

# Create features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
F = data[features]

# Create validation and training data
train_F, val_F, train_t, val_t = train_test_split(F, t, random_state=1)

# Define random forest model
rf_model = RandomForestRegressor(random_state=1)
#Fit the model with training data
rf_model.fit(train_F, train_t)

# Make predictions with validation data and calculate mean absolute error
rf_val_preds = rf_model.predict(val_F)
rf_val_mae = mean_absolute_error(rf_val_preds, val_t)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
