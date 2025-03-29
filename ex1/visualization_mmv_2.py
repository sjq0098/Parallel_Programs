import pandas as pd
import matplotlib.pyplot as plt

# 原始数据
data = {
    "File Name": ["mmv_3", "mmv_1", "mmv_2"],
    "Elapsed Time (s)": [4.153, 13.406, 14.192],
    "Clockticks (未勾选)": [16367675000, 51744615000, 54146605000],
    "Instructions Retired": [45257445000, 51007845000, 34271785000],
    "CPI Rate": [0.362, 1.014, 1.580],
    "MUX Reliability": [0.990, 0.933, 0.987]
}
df = pd.DataFrame(data)

# 按1.png、2.png、3.png顺序排序
df = df.set_index("File Name").loc[["mmv_1", "mmv_2", "mmv_3"]].reset_index()
df = df.rename(columns={'mmv_1': 'Navie', 'mmv_2': 'Navie_Uroll','mmv_3':'Cache'})

# 转换单位（亿/十亿）
df["Clockticks (未勾选)"] = df["Clockticks (未勾选)"] / 1e9
df["Instructions Retired"] = df["Instructions Retired"] / 1e9

# 可视化设置
plt.figure(figsize=(15, 10))
plt.suptitle("Performance Metrics Comparison", y=1.02, fontsize=14)

# 1. 耗时对比
plt.subplot(2, 2, 1)
plt.bar(df["File Name"], df["Elapsed Time (s)"], color='skyblue')
plt.title("Elapsed Time Comparison")
plt.ylabel("Seconds")
for i, v in enumerate(df["Elapsed Time (s)"]):
    plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')

# 2. 指令与周期对比
plt.subplot(2, 2, 2)
width = 0.4
x = range(len(df))
plt.bar(x, df["Clockticks (未勾选)"], width, label='Clockticks (Billion)', color='orange')
plt.bar([i + width for i in x], df["Instructions Retired"], width, label='Instructions (Billion)', color='green')
plt.xticks([i + width/2 for i in x], df["File Name"])
plt.title("Clockticks vs Instructions")
plt.legend()

# 3. CPI率对比
plt.subplot(2, 2, 3)
plt.bar(df["File Name"], df["CPI Rate"], color='purple')
plt.title("CPI Rate Comparison")
plt.ylabel("Cycles Per Instruction")
for i, v in enumerate(df["CPI Rate"]):
    plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')

# 4. MUX可靠性对比
plt.subplot(2, 2, 4)
plt.bar(df["File Name"], df["MUX Reliability"], color='red')
plt.title("MUX Reliability Comparison")
plt.ylabel("Reliability Score")
for i, v in enumerate(df["MUX Reliability"]):
    plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()