# TwinWorld-P0：站在巨人的肩膀上看计算机视觉 | 第一阶段最小验证

**研究目标**：验证在严格匹配的被动轨迹中，具有不同隐藏机制的世界无法区分，而向物体 A 施加相同外力后，物体 B 的响应不同。

**请注意**：这仅是物理可识别性与数据生成的 *sanity check*，**不是**新算法、科学定理或论文的最终实验。

## 运行环境
- Windows PowerShell / Python 3.11
- 仅依赖 `numpy`、`matplotlib`；无需 GPU、MuJoCo、PyTorch 或下载数据。

## 按顺序执行

在 VS Code 中打开本文件夹后，终端选择 **PowerShell**。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\01_generate.py
.\.venv\Scripts\python.exe scripts\02_validate.py
.\.venv\Scripts\python.exe scripts\03_baseline.py
.\.venv\Scripts\python.exe scripts\04_plot.py
```

若 `py -3.11` 不可用，请确认安装 Python 3.11；`py --list` 查看本机 Python。

输出：
- `data/{train,val,test}_public.npz`：模型可见观测（包含被动与干预轨迹，**不含隐藏标签**）。
- `data/{train,val,test}_private_labels.npz`：仅用于评估器的机制标签；**禁止在模型推断中读取**。
- `data/{train,val,test}_private_params.json`：生成器真值，仅供审计。
- `outputs/p0_baselines.json`：被动与主动诊断的结果。
- `outputs/p0_twinworld_comparison.png`：轨迹图。

## 必须检查的 P0 结论
1. 测试脚本的 3 项 `unittest` 通过。
2. 每个 split 的 `passive_max_abs_diff=0`；`matched_action_diff=0`。
3. 干预后两种机制的对象 B 运动存在明显差异（与 P1 传感器噪声容限是两回事）。
4. 被动 `passive chance=0.500`；主动 oracle 诊断结果由执行确定，不应把接近 1.000 当作论文创新。
5. 不修改原始测试标签与指标定义；完整保存终端日志。

## 关键实验逻辑
- `label=1`：两个物体存在隐藏的弹簧-阻尼耦合，耦合力依赖相对位移与速度。
- `label=0`：两个物体彼此独立；被动时可以产生相同的运动。
- 两世界被动初始位移、速度、摩擦项完全匹配。弹簧的静止长度等于物体间距，因此被动运动时弹簧不施力。
- 外力只作用于 A，导致耦合机制经由弹簧影响 B，独立机制不会影响 B。
- 模型输入中不存在显示弹簧、隐藏标签或生成器真值的线索。

## 发给研究导师的材料
- 全部 4 个运行命令的终端文本或截图（尤其是验证与基线）
- `outputs/p0_baselines.json`
- `outputs/p0_twinworld_comparison.png`
- Python 版本（`py -3.11 -V`）

## 后续
P1 接入 MuJoCo，仅在 P0 可识别性协议通过后开始；P2 再训练世界模型。实验升级不得悄悄修改可识别性定义与测试集。
