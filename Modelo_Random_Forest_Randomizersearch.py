import pandas as pd
import json

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

with open("data/mcc_codes.json", "r", encoding="utf-8") as f:    
    data = json.load(f)
df_mcc = pd.DataFrame.from_dict(data, orient='index')
df_mcc.reset_index(inplace=True)
df_mcc.columns = ["mcc", "descripcion"]

with open("data/train_fraud_labels.json", "r", encoding="utf-8") as f:
    data = json.load(f)
df_train_fraud_labels = pd.DataFrame.from_dict(data, orient='index')
df_train_fraud_labels = df_train_fraud_labels.T
df_train_fraud_labels.reset_index(inplace=True)
df_train_fraud_labels.columns = ["id_transaction", "target"]

df_transactions["mcc"] = df_transactions["mcc"].astype(str)
df_transactions["amount"] = (df_transactions["amount"].str.replace("$","", regex=False)).astype(float)

df2 = df_transactions.merge(df_mcc, on= 'mcc')