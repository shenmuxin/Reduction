import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b':[2, 3, 4]})
print(df)

# print(df.iloc[:, 1])
print(df.iloc[1:, [1]])
df.iloc[:, [1]] = 0
print(df)
