import pandas as pd
import os
from taipy import Gui
import seaborn as sns
import taipy.gui.builder as tgb
import matplotlib.pyplot as plt
import matplotlib
from sklearn.svm import SVR



from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("insurance.csv", index_col="index")
df = pd.DataFrame(data)
df = df.interpolate()

correlation_matrix = df[["age","bmi","charges"]].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()
sorted_corr = correlation_matrix["charges"].sort_values(ascending=False).reset_index()

# Analyze average charges based on smoker status
# print(df.groupby("smoker")["charges"].mean())
# print(df.groupby("region")["charges"].mean())

df["age_smoker_interaction"] = df["age"] * (df["smoker"] == "yes").astype(int)

# correlation_matrix_1 = df[["age_smoker_interaction","charges"]].corr()
# sns.heatmap(correlation_matrix_1, annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

# Scatter plot to visualize Age vs. Charges, differentiated by Smoker Status

# sns.scatterplot(data=df, x="age", y="charges", hue="smoker", palette="coolwarm", alpha=0.7)
# plt.title("Age vs. Charges (by Smoker Status)")
# plt.xlabel("Age")
# plt.ylabel("Charges")
# plt.legend(title="Smoker Status")
# plt.show()

df["age_bmi_interaction"] = df["age"] * df["bmi"]
# correlation_matrix_2 = df[["age_bmi_interaction","charges"]].corr()
# sns.heatmap(correlation_matrix_2, annot=True, cmap="coolwarm")
# plt.title("Correlation Matrix")
# plt.show()

df_encoded = pd.get_dummies(data=df, columns=['sex', 'region', 'smoker'], drop_first=True)

train, test = train_test_split(df_encoded, test_size=0.2, random_state=42)

x_train = train.drop(columns=["charges"])
y_train = train["charges"]
x_test = test.drop(columns=["charges"])
y_test = test["charges"]

from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
model_1 = HistGradientBoostingRegressor()

model_2 = HuberRegressor()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model_1.fit(x_train_scaled, y_train)
model_2.fit(x_train_scaled, y_train)

# Feature importance
# importances = model_1.feature_importances_
# feature_names = x_train.columns
# sorted_importances = sorted(zip(importances, feature_names), reverse=True)
# print("Feature Importances:")
# for importance, feature in sorted_importances:
#     print(f"{feature}: {importance:.2f}")


predictions_1 = model_1.predict(x_test_scaled)
predictions_2 = model_2.predict(x_test_scaled)
df_pred_1 = pd.DataFrame({"Predictions": predictions_1, "Actual": y_test})
df_pred_2 = pd.DataFrame({"Predictions": predictions_2, "Actual": y_test})

y_actual = np.array(y_test)
y_predicted_1 = np.array(predictions_1)
y_predicted_2 = np.array(predictions_2)


# Calculate MPAE
mpae_1 = np.mean(np.abs((y_actual - y_predicted_1) / y_actual)) * 100
# print(f"MPAE: {mpae_1:.2f}%")

# Calculate MPAE
mpae_2 = np.mean(np.abs((y_actual - y_predicted_2) / y_actual)) * 100
# print(f"MPAE: {mpae_2:.2f}%")


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
    user_scaled = scaler.transform(user_df)
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
     # tgb.html("h6","""A predictive model to estimate the insurance charges based on a client's attributes, such as age and health factors.\n""")

    # with tgb.layout("1 1", gap="30px"):
    #     with tgb.part():
    #         tgb.text("Following is the given data : ")
    #         tgb.table("{df.head()}")
    #     with tgb.part():
    #         tgb.text("Following is the correlation matrix of the given data : ")
    #         tgb.table("{sorted_corr}")
    #
    # with tgb.layout("1 1", gap="30px"):
    #     with tgb.part():
    #         tgb.text("Stacking Regressor Model")
    #         # tgb.table("{df_pred_1.head()}")
    #         tgb.text("MPAE error of model : {mpae_1.round(3)}%")
    #     with tgb.part():
    #         tgb.text("Huber Regressor Model")
    #         # tgb.table("{df_pred_2.head()}")
    #         tgb.text("MPAE error of model : {mpae_2.round(3)}%")
image = "logo.jpg"
page = """
<|text-center|
<|{image}|image|width=800px|height=200px|>
<h1>INSURANCE CHARGE PREDICTOR</h1>

<h6>This predictor uses hubber regressor to predict the insurance charge using various factors like user's age and BMI etc.</h6>

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
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port)
