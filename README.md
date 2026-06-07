# Climber_Vision_2026 自瞄工作空间

## 前言

本项目基于[同济大学 SuperPower 战队 25 赛季开源自瞄算法（sp_vision_25）](https://github.com/TongjiSuperPower/sp_vision_25.git)深度改进而来。


### 与 sp_vision_25 相比的主要改进

| 类别             | 改进内容                                                                                                                         |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **高低差前哨站** | 重构 3 装甲板 EKF 模型，独立估计高低板高度差；DFS 二分图槽位匹配 + 8 规则主打击面选择 + 快速重锚恢复                           |
| **能量机关模型** | 使用自训练的 YOLO_POSE 检测模型，8靶面角点+1R标角点                                                                                           |
| **大符多扇叶管理** | BigTargetGroup 5 槽扇叶跟踪与 BigTargetSelector 云台指向打分选板                                       |
| **弹道模型**     | 真空弹道统一升级为线性空气阻力模型（AirResistTrajectory），阻力系数 YAML 可配，k=0 退化为真空闭式解                             |
| **感知相机优化** | USB 感知相机改为按物理端口 udev 绑定设备名，替代锐度区分方式，并配套 camera_manager 管理工具                                 |
| **运维工具**     | 感知相机管理器、云台外参校准 GUI 工具                                                                                        |

---

## 1. 环境与依赖

### 1.1 系统环境

- **操作系统**：Ubuntu 22.04 LTS & 24.04 LTS（均测试通过）
- **运算平台**：Lenovo Baiying NUC13Plus（i7-13620H，16GB+1T）
- **主相机**：海康机器人工业相机（HikRobot MV-CA016-10UC）
- **感知相机**：杰瑞微通 USB 相机 × N（用于全向感知，720p 30Hz）
- **下位机**：RoboMaster 开发板 C 型（STM32F407）
- **IMU**：C 板内置 BMI088
- **通信方式**：MicroUSB 虚拟串口（`/dev/ttyACM0`）或 CH343P USB-TTL 模块（`/dev/ttyUSB0`）

### 1.2 基础依赖安装

```bash
sudo apt install -y \
    git \
    g++ \
    cmake \
    can-utils \
    libopencv-dev \
    libfmt-dev \
    libeigen3-dev \
    libspdlog-dev \
    libyaml-cpp-dev \
    libceres-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libusb-1.0-0-dev \
    nlohmann-json3-dev \
    openssh-server \
    screen
```

### 1.3 OpenVINO 安装

[OpenVINO 2026.1 安装指南](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-archive-linux.html)

### 1.4 Intel GPU 驱动（GPU 推理必需）

如果 OpenVINO 报 `no supported devices found`，按以下步骤安装：

```bash
# 1. 添加 Intel GPU 官方仓库签名密钥
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

# 2. 添加 Intel GPU 软件源
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu.list

# 3. 更新并安装
sudo apt update
sudo apt install -y \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    intel-media-va-driver-non-free \
    libmfx1 \
    libmfxgen1 \
    libvpl2

# 4. 添加用户到 render 和 video 组
sudo usermod -a -G render,video $USER

# 5. 重新登录，或使用 sg 命令临时运行
sg render -c "./build/infantry_mpc_debug"
```

**验证安装**：

```bash
sg render -c "clinfo -l"   # 应显示 Intel(R) OpenCL Graphics
ls -la /dev/dri/           # 检查 GPU 设备节点
```

---

## 2. 编译

```bash
cmake -B build
make -C build/ -j`nproc`
```

编译生成的可执行文件：

| 目标                                        | 说明             |
| ------------------------------------------- | ---------------- |
| `infantry`                                  | 步兵自瞄主程序   |
| `sentry` / `sentry_aim`                     | 哨兵视觉主程序   |
| `sentry_debug`                              | 哨兵调试版本     |
| `uav`                                       | 无人机视觉主程序 |
| `calibrate_camera` / `calibrate_usb_camera` | 相机内参标定     |
| `calibrate_handeye`                         | 手眼标定         |
| `split_video`                               | 视频分割工具     |
| `serial_delay`                              | 串口延迟测试     |

---

## 3. 坐标系与变换

- **相机坐标系**（OpenCV）：前方 = Z，右方 = X，下方 = Y
- **云台 FLU 坐标系**：前方 = X，左方 = Y，上方 = Z
- **手眼标定**：云台 X→相机 Z，云台 Y→相机 -X，云台 Z→相机 -Y。配置 `R_gimbal2imubody` 和 `t_camera2gimbal`
- **yaw/pitch**：基于云台 FLU 坐标系输出

> 旋转矩阵通常可靠，平移矩阵存在一定误差。外参校准可使用 `scripts/adjust_gimbal_offset.py`。

---

## 4. 相机与通信

### 4.1 主相机

海康机器人工业相机（HikRobot MV-CA016-10UC），`io/camera.cpp` 封装：

```yaml
camera_name: "hikrobot"
exposure_ms: 4000
gain: 12.0
```

### 4.2 感知相机

哨兵多路 USB 相机（杰瑞微通 720p 30Hz），按 USB 物理端口 udev 绑定设备名（`/dev/camera_front` 等），由 `scripts/perception_camera_manager.py` 统一管理。

### 4.3 串口通信

```yaml
com_port: "/dev/ttyACM0"  # 或 /dev/ttyUSB0、/dev/ttyAIM
baudrate: 921600          # 115200 或 921600
```

设备权限：`sudo usermod -a -G dialout $USER`

---

## 5. 自动启动脚本管理

项目提供 `scripts/auto_start.sh`，具备看门狗功能，程序崩溃后自动重启。

### 5.1 使用方法

```bash
./scripts/auto_start.sh start     # 启动（含看门狗守护）
./scripts/auto_start.sh stop      # 停止
./scripts/auto_start.sh status    # 查看运行状态
./scripts/auto_start.sh restart   # 重启
```

### 5.2 日志与看门狗参数

- 程序日志：`logs/infantry_5.log`
- PID 文件：`logs/infantry_5.pid`
- 默认二进制：`./build/standard`
- 默认配置：`configs/infantry_5.yaml`

看门狗可配置参数（脚本顶部）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `RUN_TIMEOUT_SEC` | 60 | 单次运行超时（秒），程序卡死即强杀；0=禁用 |
| `STARTUP_GRACE_SEC` | 5 | 首次启动前等待硬件就绪（秒） |
| `RESTART_DELAY_SEC` | 2 | 正常退出后的重启延迟（秒） |
| `QUICK_FAIL_THRESHOLD_SEC` | 3 | 运行低于此时间视为快速失败 |
| `QUICK_FAIL_MAX_COUNT` | 5 | 连续快速失败 N 次后进入长等待 |
| `LONG_WAIT_SEC` | 30 | 快速失败过多后的长等待时间（秒） |
| `DEVICES` | (空) | 需等待的设备节点，例 `/dev/ttyACM0 /dev/video0` |

---

## 6. Git 提交协议

采用 Angular Convention：`<type>(<scope>): <subject>`。

| Type | 说明 | Type | 说明 |
|------|------|------|------|
| `feat` | 新功能 | `fix` | 修复 |
| `refactor` | 重构 | `perf` | 性能优化 |
| `docs` | 文档 | `test` | 测试 |
| `chore` | 杂项 | `debug` | 调试 |



## 7. 脚本工具

### 7.1 `auto_start.sh` — 看门狗启动

目标程序的守护启动与自动重启。超时强杀（60s）、指数退避、硬件等待、HUP 信号处理。

```bash
./scripts/auto_start.sh {start|stop|status|restart}
```

> 配置参数见 [§5.2](#52-日志与看门狗参数)。

### 7.2 `perception_camera_manager.py` — 感知相机管理器

多路 USB 相机按物理端口 udev 绑定固定设备名，支持 GUI 和 CLI 两种模式。

```bash
python3 scripts/perception_camera_manager.py {ui|check}
```

### 7.3 `adjust_gimbal_offset.py` — 云台外参校准 GUI

图形化校准相机外参（Yaw/Pitch/Roll + 平移），支持微调和绝对两种模式。

```bash
python3 scripts/adjust_gimbal_offset.py
```

---

## 8. 运行测试

```bash
# 自瞄测试
./build/auto_aim_test configs/test_aim.yaml

# 能量机关测试
./build/auto_buff_test configs/test_buff.yaml

# 步兵 MPC 调试
./build/infantry_mpc_debug configs/leg_aim.yaml

# 哨兵自瞄
./build/sentry_aim configs/sentry.yaml
```

---

## 9. 文件结构

```
Climber_Vision_2026
├── assets/              # 模型权重等资源文件
├── calibration/         # 相机内参标定、手眼标定、串口延迟测试
├── configs/             # 各兵种 YAML 配置文件
├── io/                  # 硬件抽象层
│   ├── camera.cpp       #   海康工业相机封装
│   ├── hikrobot/        #   海康机器人 SDK 封装
│   ├── usbcamera/       #   USB 感知相机驱动（按端口绑定设备名）
│   ├── gimbal/          #   云台串口通信
│   ├── cboard.*         #   C 板串口通信协议
│   └── serial/          #   串口底层库
├── patterns/            # 能量机关 R 标图案模板
├── records/             # 录制帧数据（时间戳+四元数+图像）
├── scripts/             # 运维脚本
│   ├── auto_start.sh                # 看门狗启动脚本
│   ├── adjust_gimbal_offset.py      # 云台外参校准 GUI
│   └── perception_camera_manager.py # USB 感知相机管理器（udev 端口绑定）
├── src/                 # 应用层（infantry / sentry / uav 主程序）
├── tasks/               # 功能层
│   ├── auto_aim/        #   自瞄算法（YOLO、解算、跟踪、瞄准、射击、MPC规划）
│   ├── auto_buff/       #   能量机关算法（R标检测、扇叶识别、EKF预测、MPC控制）
│   └── omniperception/  #   全向感知算法（多方向感知、装甲板选择、调试视图）
├── test/                # 独立测试程序（17个）
├── tools/               # 工具层（EKF、数学工具、日志、图像工具、轨迹解算、录制器）
├── CMakeLists.txt       # CMake 构建配置
└── README.md
```

---

## 参考与致谢

- 深度参考 [同济大学 SuperPower 战队 25 赛季开源（sp_vision_25）](https://github.com/TongjiSuperPower/sp_vision_25.git)
- 感谢同济开源项目的理论框架与代码基础
