import pandas as pd
import os
from taipy import Gui
import taipy.gui.builder as tgb
import matplotlib.pyplot as plt
import matplotlib
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def evaluate_model_performance(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return rmse, r2, mae
    # print(f"MAE: {mae}")
    # print(f"MSE: {mse}")
    # print(f"RMSE: {rmse}")
    # print(f"R²: {r2:.2f}")

from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("insurance.csv", index_col="index")
df = pd.DataFrame(data)
df = df.interpolate()

correlation_matrix = df[["age","bmi","charges"]].corr()

sorted_corr = correlation_matrix["charges"].sort_values(ascending=False).reset_index()

df["age_smoker_interaction"] = df["age"] * (df["smoker"] == "yes").astype(int)



df["age_bmi_interaction"] = df["age"] * df["bmi"]

df_encoded = pd.get_dummies(data=df, columns=['sex', 'region', 'smoker'], drop_first=True)

df["log_charges"] = np.log(df["charges"])

train, test = train_test_split(df_encoded, test_size=0.2, random_state=42)

x_train = train.drop(columns=["charges"])
y_train = train["charges"]
x_test = test.drop(columns=["charges"])
y_test = test["charges"]

from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_train_poly = poly.fit_transform(x_train_scaled)
x_test_poly = poly.transform(x_test_scaled)


import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
model_1 = HistGradientBoostingRegressor()

model_2 = HuberRegressor()


# Train XGBoost Regressor with default parameters (can also be tuned)
xgb_model = XGBRegressor(n_estimators=850, learning_rate=0.008, max_depth=16, colsample_bytree=0.9, objective='reg:tweedie')
# xgb_model.fit(x_train_poly, y_train_scaled.ravel())

# model_1.fit(x_train_scaled, y_train_scaled.ravel())  # Use ravel() to flatten for regression
# model_2.fit(x_train_scaled, y_train_scaled.ravel())

#Ensembling using Voting Regressor
# ensemble_model = VotingRegressor(estimators=[
#     # ('hist', model_1),
#     # ('hub', model_2),
#     ('xgb', xgb_model)
# ])

# ensemble_model.fit(x_train_poly, y_train_scaled .ravel())

import joblib

# Save the trained ensemble model
# joblib.dump(ensemble_model, "ensemble_model.pkl")
ensemble_model = joblib.load("ensemble_model.pkl")
# Predictions
ensemble_predictions = ensemble_model.predict(x_test_poly)
ensemble_predictions = scaler_y.inverse_transform(ensemble_predictions.reshape(-1, 1))

# Model Performance
rmse, r2, mae=evaluate_model_performance(ensemble_predictions, y_test.to_numpy().reshape(-1, 1))

#User Variables:
user_age = 0
user_bmi = 0
user_sex = ""
user_region = ""
user_smoker = ""
user_children = 0


def preprocess_input(age, bmi, sex, region, smoker,children):
    # Convert categorical variables into one-hot encoding
    user_data = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "age_smoker_interaction": age * (smoker == "yes"),
        "age_bmi_interaction": age * bmi,
        "sex_male": 1 if sex == "male" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
    }

    # Convert to DataFrame
    user_df = pd.DataFrame([user_data])

    # Scale the inputs
    user_scaled = scaler_y.transform(user_df)
    return user_scaled

predicted_charge = 0

def reset(state):
    state.user_age = 0
    state.user_bmi = 0
    state.user_sex = ""
    state.user_region = ""
    state.user_smoker = ""
    state.user_children = 0
    state.predicted_charge = 0
def predict_insurance(state):
    user_scaled = preprocess_input(state.user_age, state.user_bmi, state.user_sex, state.user_region, state.user_smoker, state.user_children)
    state.predicted_charge = (model_2.predict(user_scaled)[0]).round(2)

image = "logo.jpg"
page = """
<|text-center|
<|{image}|image|width=800px|height=200px|>
<h1>INSURANCE CHARGE PREDICTOR</h1>

<h6>This predictor uses XGBoost regressor to predict the insurance charge using various factors like user's age and BMI etc.</h6>
<|layout|columns = 1 1 1 1|
# MAE : <|metric|value={mae:.2f}|>
# RMSE : <|metric|value={rmse:.2f}|>
# R² : <|metric|value={r2:.2f}|>
|>
## <|Enter your details|>

## <|Enter your age : |><|{user_age}|number|>

## <|Enter your BMI : |> <|{user_bmi}|number|>

## <|Enter your Gender : |> <|{user_sex}|selector|lov=male;female|>

## <|Enter your Region : |><|{user_region}|selector|lov=northeast; northwest; southeast; southwest|>

## <|Do you smoke? :|> <|{user_smoker}|selector|lov=yes;no|>

Number of children : <|{user_children}|number|width = 600px|>

<|Submit|button|on_action=predict_insurance|width = 500px|>
<|Reset|button|on_action=reset|width = 500px|>

## Predicted Insurance Charge:
<|${predicted_charge}|text|width=100px|height=200px|>
>
"""

if __name__== "__main__":
    app = Gui(page)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
