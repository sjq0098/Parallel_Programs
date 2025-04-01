import pandas as pd
import matplotlib.pyplot as plt
import re

# 原始数据（已处理标记和缺失值）
data = {
    "File Name": ["4.png", "5.png", "6.png"],
    "Elapsed Time (s)": [0.110, 0.105, 0.084],
    "Clockticks (未勾选)": [263_560_000, 344_425_000, 230_615_000],
    "Instructions Retired": [224_625_000, 344_425_000, 209_650_000],
    "CPI Rate": [1.173, 1.000, 1.100]
}

df = pd.DataFrame(data)
df["File Name"] = df["File Name"].replace({
    "4.png": "Navie",
    "5.png": "Pair",
    "6.png": "Navie_Unroll"
})

# 单位转换（百万）
df["Clockticks (未勾选)"] = df["Clockticks (未勾选)"] / 1e6
df["Instructions Retired"] = df["Instructions Retired"] / 1e6

# 可视化设置
plt.figure(figsize=(14, 8))
plt.suptitle("Performance Metrics Comparison (Short Tasks)", y=1.02, fontsize=14)

# 1. 耗时对比
plt.subplot(2, 2, 1)
bars = plt.bar(df["File Name"], df["Elapsed Time (s)"], color='#1f77b4')
plt.title("A. Execution Time", pad=12)
plt.ylabel("Seconds")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, 
             f"{height:.3f}", ha='center', va='bottom')

# 2. 时钟周期与指令对比
plt.subplot(2, 2, 2)
width = 0.35
x = range(len(df))
plt.bar(x, df["Clockticks (未勾选)"], width, label='Clockticks (Million)', color='#ff7f0e')
plt.bar([i + width for i in x], df["Instructions Retired"], width, 
        label='Instructions (Million)', color='#2ca02c')
plt.xticks([i + width/2 for i in x], df["File Name"])
plt.title("B. Hardware Utilization", pad=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 3. CPI率对比
plt.subplot(2, 2, 3)
bars = plt.bar(df["File Name"], df["CPI Rate"], color='#9467bd')
plt.title("C. CPI Efficiency", pad=12)
plt.ylabel("Cycles Per Instruction")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, 
             f"{height:.3f}", ha='center', va='bottom')

# 4. 综合效率分析
plt.subplot(2, 2, 4)
plt.axis('off')

plt.tight_layout()
plt.show()