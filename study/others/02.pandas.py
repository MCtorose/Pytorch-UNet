import pandas as pd

dataPatch = r"E:\Desktop\Python\数据分析\boss.csv"
# 导入csv
csv = pd.read_csv(dataPatch, encoding='utf-8')
print(csv)
# 导入excel
# excel = pd.read_excel(dataPatch, engine="openpyxl")
dataPatch = r"E:\Desktop\new_net\study\01"
txt = pd.read_csv(dataPatch, sep='\t', encoding='utf-8')
print(txt)

print(csv.head(5))  # 前5条
print(csv.iloc[5])  # 索引第五的数据  中括号
print(csv.iloc[[0, 5]])  # 0-5行的数据
print(csv.iloc[0:5])  # 0-5行的数据
# print(csv[["表头名称"]])

# print(csv.loc[() & ()])

# 缺失值填充
# csv["摄政"] = csv["摄政"].fillna(29)

# csv.sort_values(by="出场次数", ascending=False)  # 根据出场次数，降序

# 求和
data = [[110, 105, 99], [105, 88, 115], [109, 120, 134]]
index = [i for i in range(1, 4)]
columns = ["A", "B", "C"]

df = pd.DataFrame(data=data, index=index, columns=columns)
df["总成绩"] = df.sum(axis=1)
print(df)

new = df.max()
df = df._append(new, ignore_index=True)
print(df)

# 合并两张表

df1 = pd.DataFrame({
    'id': ['1', '2', '3'],
    'chinese': [110, 120, 230],
    'chi': [110, 120, 230]
})
df2 = pd.DataFrame({
    'id': ['1', '2', '3'],
    'english': [110, 120, 230],
    'chi': [110, 120, 230]
})

df3 = pd.merge(df1, df2, on="id")
print(df3)


# 导出文件

df3.to_csv("./123.csv")