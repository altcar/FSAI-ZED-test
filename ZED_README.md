# ZED SDK 虚拟双目实验：需要 Windows 实机验证

这条路径的目标是一次比较：UE 同步左右图 → ZED SDK 深度；同一张 SDK 左图 → Depth Net；二者与相同视角的 UE Z 真值比较。普通 ZED SDK 相机/SVO入口不能直接把两张 PNG 当成相机，本包增加了官方仿真流 C 接口的 Python 桥接。

代码以 Stereolabs 官方 [zed-isaac-sim](https://github.com/stereolabs/zed-isaac-sim/tree/f9160973f321ec743f626188244cd9702d85bec6) 的 ABI/标定格式为依据；没有启动 Isaac Sim，也不需要接收方安装 UE。这里的 Python 桥接是新写的实验代码，不是 Stereolabs 已验证的成品工具。

**本机是 Mac，尚未验证 DLL 加载、Windows SDK 接收、CUDA 推理和 SDK 标定读取。不能保证双击 04 就成功。** 失败时保留终端错误、`outputs/environment.json`、`outputs/sdk_three_way/streamer.log`。现有 `reference_results` 和 `validation` 中没有 SDK 深度或三方成功结果。

## 安装与运行

1. 先完成主 README 的 00–02，确认 PyTorch 可以在 GPU 0 推理。
2. 安装 [ZED SDK 5.5.x Windows 10/11、CUDA 12](https://www.stereolabs.com/developers/release) 版本，使用安装向导要求的 CUDA/驱动。建议当前 5.5.1。安装及 SDK 模型下载由接收方按官方许可完成，本包不包含这些安装包和设备专用优化模型。
3. 重启后双击 `03_install_zed_python.cmd`，使用 SDK 自带 `get_python_api.py`，把与 SDK 匹配的 pyzed 安装到本包 `.venv`。若 SDK 安装目录不同，设置 `ZED_SDK_ROOT_DIR`。
4. 双击 `04_run_zed_three_way_EXPERIMENTAL.cmd`。脚本优先尝试 `zed_runtime/sl_zed64.dll`，其次尝试 SDK `bin` 中的 DLL。
5. 若 DLL 缺少仿真接口，可运行 `03b_download_sim_runtime_IF_NEEDED.cmd`，从官方 CDN 获取该官方仓库 build.bat 指定的独立仿真运行库，再试 04。

下载器使用的官方 URL 保存在 `scripts/download_sim_runtime.py`。官方仓库把该依赖独立标记为 **5.2.0**，而同一仓库的新版头文件要求 SDK C ABI **5.5**。这两个版本标记能否对应，不能在 Mac 上确认；因此下载不代表已兼容。桥接会检查 DLL 实际报告的 SDK 版本和导出符号，只接受 5.5.x。若检查拒绝，应向 Stereolabs 获取与该 ABI 对应的运行库，不能通过删掉检查来强行运行，也不能以其他立体匹配算法冒充 SDK。

可显式指定已获得的运行库：

```bat
.venv\Scripts\python.exe scripts\run_sdk.py --dll "D:\MatchingRuntime\sl_zed64.dll"
```

实际需要 `getZEDSDKRuntimeVersion_C`、`init_streamer`、`stream_rgb`、`is_sn_valid`、`close_streamer`。普通 SDK DLL 未必提供这些仿真导出。附带 `licenses/types_c.h`、`zed_interface_loader.hpp` 作为 ABI 依据。

## 实验采用的相机

原始 UE 数据来自原版 ZED SN2448592 的工厂标定，左右眼有微小旋转和偏移。为了符合官方虚拟相机模型，本实验先用 OpenCV 校正这对图，生成平行双目：960×600，fx=fy=636.25，cx=480，cy=300，基线约 0.119724480 m；对应官方虚拟 ZED X One GS 窄镜头参数。

因此这是一场景的 SDK 虚拟双目测试，**不能当作实车原版 ZED + 项目原用 SDK 5.0.7 的精确复现**。重采样、分辨率、虚拟相机型号和 SDK 版本均有变化。

UE Z 随左相机的校正旋转重新投影，并转换成校正坐标系的光轴 Z，采用最近邻深度采样，避免把前后景深度混成一个值。物体边缘仍有约一像素采样误差。标定和变换保存在每次生成的 `bridge_*/geometry.json` 中。

程序用预留的虚拟序列号 `110307008` 创建配置。接收端使用本包内的独立配置目录；原生流接口没有配置路径参数，故发送端还需要临时写 `C:\ProgramData\Stereolabs\settings\SN110307008.conf`。若此文件已经存在会停止，绝不覆盖；正常结束时删除本次创建的文件。权限不足时会明确报错。程序异常被系统强制关闭可能留下该虚拟文件，确认后可手动移除再运行；不要改动真实序列号配置。

SDK 打开后会检查它实际读取的原始内参、畸变、基线和旋转；不匹配会停止。接收后的 rectified K 用于生成最终 UE 真值，因此不是单纯把数组改成一样大小。

## 数据流和输出

发送进程只读取左右 RGB，`stream_depth=false`，不读取 UE 真值。采用 H265、16000kbps、本机 RTP、15fps，反复播放同一静态帧。它包含有损编码的影响，不代表实时车辆运动测试。

SDK 默认 `NEURAL`、米制 `IMAGE` 坐标系，关闭 self-calibration 和 depth stabilization，等待 10 次成功 grab。只运行深度，不启用定位或物体检测。没有真实传感器 IMU；流内使用静止占位姿态。这里比较的不是 IMU 或定位能力。

SDK 的 `VIEW.LEFT` 和 `MEASURE.DEPTH` 从同一次 grab 取出，保存后关闭 SDK 和流进程，再运行 Depth Net，减少同时占用显存。最终网络输入是实际接收、解码和校正后的 `sdk_left.png`。

| 输出 | 内容 |
|---|---|
| `sdk_left.png` | SDK 接收的左图，也是网络实际输入 |
| `sdk_z_m.npy` | SDK 从双目图估计的 float32 光轴 Z，米 |
| `ue_z_m.npy` | 对齐到 SDK 左图/标定的 UE 几何 Z 真值 |
| `sdk_capture.json` | SDK 版本、参数、K、基线、时间戳、几何和编码信息 |
| `comparison/depthnet_z_m.npy` | 网络在这张 SDK 左图上的原始深度输出 |
| `comparison/summary.json` | 每方法有效率、误差以及共同有效像素上的比较 |
| `comparison/comparison.png` | UE、SDK、Depth Net 和误差可视化 |
| `streamer.log` | 原生仿真流状态与错误 |

若显存不足，先关闭其他 GPU 程序，保留顺序运行设置。可改为 SDK 非神经模式：

```bat
.venv\Scripts\python.exe scripts\run_sdk.py --depth-mode PERFORMANCE
```

此时务必在报告里注明模式发生变化；不得把 PERFORMANCE 结果标成 NEURAL。不同模式是否被当前 SDK 接受，以实际运行结果为准。

SDK 首次可能联网下载并优化其模型。不要因第一次 `open()` 较慢就把参考数组复制成“SDK 结果”。成功三方输出应有真实 `sdk_capture.json`，并且 `comparison/summary.json` 中 `three_way_complete` 为 true。
