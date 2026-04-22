# 四足机器人动作重定向与可视化工具

这个仓库现在保留的是一个精简后的动作处理工具链，主要用于：

- 读取原始四足动作
- 将 Laikago 风格动作重定向到 `a1`
- 将 Laikago 风格动作重定向到 `go2`
- 将 `19dof` 动作转换成下游数据集常用的 `61dof` 格式
- 用 PyBullet 直接检查动作结果

原来上游项目中的训练栈、MPC 控制器、第三方求解器和大体量依赖已经移除。当前仓库只保留动作生成、动作转换、动作检查所需的最小代码和本地机器人资产。

## 1. 仓库目录

- `retarget_motion/`
  - 机器人配置
  - 动作重定向脚本
  - `19dof -> 61dof` 转换脚本
  - PyBullet 可视化脚本
- `motion_imitation/`
  - 当前动作处理仍然依赖的最小机器人与工具模块
- `assets/robots/`
  - `a1`、`go2`、`sizu`、`laikago` 的本地资产
- `motion_imitation/data/motions/`
  - 源动作库
- `motion_imitation/data/motions_a1/`
  - 生成后的 A1 `19dof` 动作
- `motion_imitation/data/motions_a1_61dof/`
  - 生成后的 A1 `61dof` 动作
- `motion_imitation/data/motions_go2/`
  - 生成后的 Go2 `19dof` 动作
- `motion_imitation/data/motions_go2_amp49/`
  - 从 `go2_flip_TO` 引入的 Go2 高难动作 `49dof` AMP 源文件
- `motion_imitation/data/motions_go2_61dof/`
  - 生成后的 Go2 `61dof` 动作

## 2. 环境准备

当前脚本已经在下面这个 conda 环境里验证过：

```bash
source /home/shibo/anaconda3/etc/profile.d/conda.sh
conda activate unitree-rl
```

如果你每次都要执行，可以直接先运行：

```bash
source /home/shibo/anaconda3/etc/profile.d/conda.sh && conda activate unitree-rl
```

仓库剩余代码的最小 Python 依赖见 `requirements.txt`。

## 3. 支持的动作格式

### 3.1 19dof

每一帧的格式是：

- 根位置 `root_pos(3)`
- 根四元数 `root_rot(4)`
- 12 个关节角 `joint_pose(12)`

总长度为 `19`。

### 3.2 61dof

每一帧的格式是：

- `pose(19)`
- 足端局部位置 `toe_local_pos(12)`
- 根线速度 `root_lin_vel(3)`
- 根角速度 `root_ang_vel(3)`
- 关节速度 `joint_vel(12)`
- 足端局部速度 `toe_local_vel(12)`

总长度为 `61`。

### 3.3 外部 Go2 agile AMP 49dof

从 `go2_flip_TO` 引入的外部 Go2 高难动作，每一帧格式是：

- `pose(19)`
- 足端局部位置 `toe_local_pos(12)`
- 根线速度 `root_lin_vel(3)`
- 根角速度 `root_ang_vel(3)`
- 关节速度 `joint_vel(12)`

总长度为 `49`。

它和当前仓库 `61dof` 的差别只有最后的 `toe_local_vel(12)`。当前仓库会在导入时用相邻帧差分自动补出来。

## 4. 当前动作库

当前 `a1` 和 `go2` 使用的是同一套动作名：

- `dog_pace`
- `dog_trot`
- `dog_backwards_pace`
- `dog_backwards_trot`
- `dog_spin`
- `hopturn`
- `inplace_steps`
- `runningman`
- `sidesteps`
- `turn`

其中：

- `dog_pace`、`dog_trot` 是原始关键点重定向得到
- `dog_backwards_pace`、`dog_backwards_trot` 是派生动作
- 其余几个动作走的是从已有动作文件适配到目标机器人的流程

额外支持的 Go2 高难 agile 动作：

- `quad_backflip`
- `quad_sideflip`
- `quad_jump_forward_1m`

## 5. 常用脚本

### 5.1 生成 A1 动作库

脚本：

- `retarget_motion/generate_a1_motion_library.py`

作用：

- 生成 A1 `19dof` 动作
- 输出到 `motion_imitation/data/motions_a1/`
- 同时写入 `provenance.json`

命令：

```bash
python retarget_motion/generate_a1_motion_library.py \
  --output_dir motion_imitation/data/motions_a1
```

只生成某几个动作：

```bash
python retarget_motion/generate_a1_motion_library.py \
  --output_dir motion_imitation/data/motions_a1 \
  --motion dog_pace \
  --motion dog_trot
```

### 5.2 生成 Go2 动作库

脚本：

- `retarget_motion/generate_go2_motion_library.py`

作用：

- 生成 Go2 `19dof` 动作
- 输出到 `motion_imitation/data/motions_go2/`
- 同时写入 `provenance.json`

命令：

```bash
python retarget_motion/generate_go2_motion_library.py \
  --output_dir motion_imitation/data/motions_go2
```

只生成某几个动作：

```bash
python retarget_motion/generate_go2_motion_library.py \
  --output_dir motion_imitation/data/motions_go2 \
  --motion hopturn \
  --motion runningman \
  --motion sidesteps
```

### 5.3 将 19dof 转成 61dof

脚本：

- `retarget_motion/convert_19dof_to_61dof.py`

作用：

- 将 `19dof` 动作转换成 `61dof`
- 支持 `a1`、`go2`、`laikago`、`vision60`

#### A1 全量转换

```bash
python retarget_motion/convert_19dof_to_61dof.py \
  --input_dir motion_imitation/data/motions_a1 \
  --output_dir motion_imitation/data/motions_a1_61dof \
  --robot a1 \
  --motion dog_pace \
  --motion dog_trot \
  --motion dog_backwards_pace \
  --motion dog_backwards_trot \
  --motion dog_spin \
  --motion hopturn \
  --motion inplace_steps \
  --motion runningman \
  --motion sidesteps \
  --motion turn \
  --motion_weight 1.0
```

#### Go2 全量转换

```bash
python retarget_motion/convert_19dof_to_61dof.py \
  --input_dir motion_imitation/data/motions_go2 \
  --output_dir motion_imitation/data/motions_go2_61dof \
  --robot go2 \
  --motion dog_pace \
  --motion dog_trot \
  --motion dog_backwards_pace \
  --motion dog_backwards_trot \
  --motion dog_spin \
  --motion hopturn \
  --motion inplace_steps \
  --motion runningman \
  --motion sidesteps \
  --motion turn \
  --motion_weight 1.0
```

#### 单动作转换

例如只转换 Go2 的 `runningman`：

```bash
python retarget_motion/convert_19dof_to_61dof.py \
  --input_dir motion_imitation/data/motions_go2 \
  --output_dir motion_imitation/data/motions_go2_61dof \
  --robot go2 \
  --motion runningman \
  --motion_weight 1.0
```

### 5.4 将外部 Go2 agile 49dof 转成 61dof

脚本：

- `retarget_motion/convert_amp_49dof_to_61dof.py`

作用：

- 将外部 `49dof` AMP 动作转换成当前仓库使用的 `61dof`
- 当前用于 `go2_flip_TO` 导出的 Go2 高难动作

例如只转换一个后空翻动作：

```bash
python retarget_motion/convert_amp_49dof_to_61dof.py \
  --input_dir motion_imitation/data/motions_go2_amp49 \
  --output_dir motion_imitation/data/motions_go2_61dof \
  --robot go2 \
  --motion quad_backflip
```

### 5.5 直接导入 Go2 agile 动作库

脚本：

- `retarget_motion/import_go2_agile_motions.py`

作用：

- 直接把已经放进仓库的 Go2 高难 `49dof` AMP 动作导入到 `motions_go2_61dof`
- 同时更新 `provenance.json`

导入全部当前 agile 动作：

```bash
python retarget_motion/import_go2_agile_motions.py \
  --output_dir motion_imitation/data/motions_go2_61dof
```

只导入后空翻和侧空翻：

```bash
python retarget_motion/import_go2_agile_motions.py \
  --output_dir motion_imitation/data/motions_go2_61dof \
  --motion quad_backflip \
  --motion quad_sideflip
```

## 6. 推荐使用流程

### 6.1 A1 原始动作到 61dof 的完整流程

```bash
source /home/shibo/anaconda3/etc/profile.d/conda.sh && conda activate unitree-rl
python retarget_motion/generate_a1_motion_library.py \
  --output_dir motion_imitation/data/motions_a1
python retarget_motion/convert_19dof_to_61dof.py \
  --input_dir motion_imitation/data/motions_a1 \
  --output_dir motion_imitation/data/motions_a1_61dof \
  --robot a1 \
  --motion dog_pace \
  --motion dog_trot \
  --motion dog_backwards_pace \
  --motion dog_backwards_trot \
  --motion dog_spin \
  --motion hopturn \
  --motion inplace_steps \
  --motion runningman \
  --motion sidesteps \
  --motion turn \
  --motion_weight 1.0
```

### 6.2 Go2 原始动作到 61dof 的完整流程

```bash
source /home/shibo/anaconda3/etc/profile.d/conda.sh && conda activate unitree-rl
python retarget_motion/generate_go2_motion_library.py \
  --output_dir motion_imitation/data/motions_go2
python retarget_motion/convert_19dof_to_61dof.py \
  --input_dir motion_imitation/data/motions_go2 \
  --output_dir motion_imitation/data/motions_go2_61dof \
  --robot go2 \
  --motion dog_pace \
  --motion dog_trot \
  --motion dog_backwards_pace \
  --motion dog_backwards_trot \
  --motion dog_spin \
  --motion hopturn \
  --motion inplace_steps \
  --motion runningman \
  --motion sidesteps \
  --motion turn \
  --motion_weight 1.0
```

### 6.3 Go2 高难 agile 动作导入到 61dof 的流程

```bash
source /home/shibo/anaconda3/etc/profile.d/conda.sh && conda activate unitree-rl
python retarget_motion/import_go2_agile_motions.py \
  --output_dir motion_imitation/data/motions_go2_61dof \
  --motion quad_backflip \
  --motion quad_sideflip \
  --motion quad_jump_forward_1m
```

## 7. PyBullet 可视化

脚本：

- `retarget_motion/view_motion_pybullet.py`

支持：

- 输入单个动作文件
- 输入动作目录
- 支持机器人：`a1`、`go2`、`sizu`、`laikago`

### 7.1 看整个 A1 动作目录

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_a1 \
  --robot a1
```

### 7.2 看整个 Go2 动作目录

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_go2 \
  --robot go2
```

### 7.3 看 Go2 的 61dof 动作目录

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_go2_61dof \
  --robot go2
```

### 7.4 看单个动作

例如检查 Go2 的 `hopturn`：

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_go2/hopturn.txt \
  --robot go2
```

例如检查 Go2 的 `61dof runningman`：

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_go2_61dof/runningman.txt \
  --robot go2
```

### 7.5 常用可视化参数

加快播放速度：

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_go2 \
  --robot go2 \
  --playback_speed 1.5
```

只播放 1 轮：

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_go2 \
  --robot go2 \
  --loops 1
```

关闭足端 marker：

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_go2_61dof \
  --robot go2 \
  --hide_toe_markers
```

关闭速度箭头：

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_go2_61dof \
  --robot go2 \
  --hide_velocity
```

关闭跟随相机：

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_go2 \
  --robot go2 \
  --no_follow_camera
```

关闭地面：

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_go2 \
  --robot go2 \
  --no_ground
```

## 8. Go2 相关说明

### 8.1 关节导出顺序

Go2 动作文件导出顺序已经对齐到外部 AMP 使用顺序：

- `FR`
- `FL`
- `RR`
- `RL`

内部 PyBullet 关节 `q` 顺序与导出顺序不同，这一层转换已经在代码里处理，不需要手动改文件顺序。

### 8.2 小腿分支求解

Go2 在适配已有动作文件时，已经改成使用解析腿解，而不是 Bullet 的通用 IK 分支。这样可以避免之前 `calf` 解翻到错误分支导致的动作异常。

### 8.3 足底轻微插地

部分动作本身就不是严格零穿透，尤其是 `hopturn`、`runningman`、`sidesteps` 这类幅度较大的动作。当前 Go2 适配路径已经加入了额外的根高度补偿：

- `retarget_motion/retarget_config_go2.py`
- 参数名：`ADAPT_ROOT_HEIGHT_OFFSET`

如果后续还想继续微调足底高度，优先调这个值，不要先去乱改其他参数。当前默认值是：

```python
ADAPT_ROOT_HEIGHT_OFFSET = 0.02
```

如果你觉得还偶发插地，可以继续往上试：

- `0.025`
- `0.03`

调完后重新执行 Go2 的生成和转换命令即可。

## 9. 输出文件说明

生成动作后，通常会看到这些文件：

- `*.txt`：动作文件
- `provenance.json`：每个动作的来源信息

`provenance.json` 里会记录：

- 动作是 `raw`、`derived` 还是 `adapted`
- 原始输入文件路径
- 输出文件路径

## 10. 已知说明

- A1 的 URDF 在 PyBullet 里会打印一部分固定链接缺少惯量的 warning，这些 warning 不影响当前动作生成流程。
- `motion_imitation/robots/a1.py` 里保留了一些历史代码路径，导入时可能会看到 `arccos` 相关 warning；当前动作生成结果已经单独验证过。
- 可视化脚本现在只依赖仓库内部机器人资产，不再依赖外部 `legged_gym` 或 IsaacGymLoco 目录。

## 11. 许可证

仓库中仍然包含来自原始项目的部分代码和资产，请同时参考：

- `LICENSE.txt`
- `assets/` 下各资产自带的许可证文件
