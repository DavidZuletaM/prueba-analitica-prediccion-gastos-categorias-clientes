import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Creación de los Dataframes
df_transactions = pd.read_csv("data/transactions_data.csv",sep=",")
df_transactions.to_parquet("data/transactions_data.parquet", index=False)
df_transactions = pd.read_parquet("data/transactions_data.parquet")

df_users_data = pd.read_csv("data/users_data.csv", sep=",")
df_users_data.to_parquet("data/users_data.parquet", index=False)
df_users_data = pd.read_parquet("data/users_data.parquet")

# Creación del Dataset para entrenar y testear el modelo de Machine Learning
df_union = (
    df_transactions
    .merge(
        df_users_data[["id","per_capita_income","yearly_income",
                       "credit_score","current_age"]],
        right_on="id",
        left_on="client_id",
        how = 'left'
    )
)

df_union["amount"] = (df_union["amount"].str.replace("$","", regex=False)).astype(float)
df_union["per_capita_income"] = (df_union["per_capita_income"].str.replace("$","", regex=False)).astype(int)
df_union["yearly_income"] = (df_union["yearly_income"].str.replace("$","", regex=False)).astype(int)

df_resultado = df_union.groupby("mcc").agg({
        "id_x":'count',"amount":'sum',"card_id":'nunique',
        "zip":'nunique',"client_id":'nunique',"merchant_id":'nunique',
        "per_capita_income":'mean',"yearly_income":'mean',
        "credit_score":'mean',"current_age":'mean'
    }).reset_index()

#Definición de variable respuesta
y = df_resultado['amount']
X = df_resultado.copy()
X.drop('amount', axis=1, inplace=True)

#Preparación de los conjuntos de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Pipeline
n_pipeline = Pipeline(
    steps=[
        (
            "column_transformer",
            make_column_transformer(
                (
                    MinMaxScaler(feature_range= (0,1)),
                    make_column_selector(dtype_include=['float64','int64']),
                ),
                (
                    PolynomialFeatures(interaction_only=True, include_bias=False),
                    make_column_selector(dtype_include=[np.number]),   
                ),
            ),
        ),
        (
            "feature_selection",
            SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=0)),
        ),
        (
            "RFRegressor",
            RandomForestRegressor(random_state= 0),
        ),
    ],
    verbose=True,
)

# Creación de grilla de hiperparámetros
param_grid = {
    'RFRegressor__n_estimators': [100, 150],
    'RFRegressor__criterion': ['squared_error'],
    'RFRegressor__max_depth': [15,20,None]
}

# Busqueda de los hiperparámetros
rf_gridsearch = GridSearchCV(
    n_pipeline,
    param_grid,
    cv=7,
    refit=True,
    verbose=2
)

# Entrenamiento del modelo
rf_gridsearch.fit(X_train,y_train)

# Generar resultados de predicciones
y_pred = rf_gridsearch.best_estimator_.predict(X_test)

# Cálculo de Métricas
mse= mean_squared_error(y_test, y_pred)
print(mse)
r2= r2_score(y_test,y_pred)
print(r2)