import pandas as pd

df1 = pd.read_csv("data/breast-cancer.csv")
df2 = pd.read_csv("data/data.csv")

print("Shape 1:", df1.shape)
print("Shape 2:", df2.shape)

# Compare columns safely
print("\nColumns in breast-cancer.csv but not in data.csv:")
print(set(df1.columns) - set(df2.columns))

print("\nColumns in data.csv but not in breast-cancer.csv:")
print(set(df2.columns) - set(df1.columns))
