import pandas as pd
from sklearn.model_selection import train_test_split

# Creación de los Dataframes
df_transactions = pd.read_csv("data/transactions_data.csv",sep=",")
df_transactions.to_parquet("data/transactions_data.parquet", index=False)
df_transactions = pd.read_parquet("data/transactions_data.parquet")

df_cards_data = pd.read_csv("data/cards_data.csv", sep=",")
df_cards_data.to_parquet("data/cards_data.parquet", index=False)
df_cards_data= pd.read_parquet("data/cards_data.parquet")

df_users_data = pd.read_csv("data/users_data.csv", sep=",")
df_users_data.to_parquet("data/users_data.parquet", index=False)
df_users_data = pd.read_parquet("data/users_data.parquet")

# Creación del Dataset para entrenar y testear el modelo de Machine Learning
df_union = (
    df_transactions
    .merge(
        df_cards_data[["id","credit_limit"]], 
        right_on="id",
        left_on="card_id",
        how = 'left'
    )
    .merge(
        df_users_data[["id","per_capita_income","yearly_income","credit_score","current_age"]],
        right_on="id",
        left_on="client_id",
        how = 'left'
    )
)

df_union["amount"] = (df_union["amount"].str.replace("$","", regex=False)).astype(float)
df_union["credit_limit"] = (df_union["credit_limit"].str.replace("$","", regex=False)).astype(int)
df_union["per_capita_income"] = (df_union["per_capita_income"].str.replace("$","", regex=False)).astype(int)
df_union["yearly_income"] = (df_union["yearly_income"].str.replace("$","", regex=False)).astype(int)

df_resultado = df_union.groupby("mcc").agg({
        "id":'count',"amount":'sum',"card_id":'nunique',
        "zip":'nunique',"client_id":'nunique',"merchant_id":'nunique',
        "credit_limit":'mean',"per_capita_income":'mean',
        "yearly_income":'mean',"credit_score":'mean',"current_age":'mean'
    }).reset_index()

#Definición de variable respuesta
y = df_resultado['amount']
X = df_resultado.copy()
X.drop('amount', axis=1, inplace=True)

#Preparación de los conjuntos de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Pipeline
