# FSAI 深度比较包：Windows 10/11 + RTX 3070 8GB

本包用于把一次 UE 采集带到另一台电脑上，运行 FSAI-Seg 的单目深度，并与 UE 光轴 Z 深度真值比较。包含模型、对应版本的推理源码、单目样例、双目样例、真值、标定、Windows 入口及本机验证结果。接收方无需安装 UE 或复制原项目。

RTX 3070 8GB 满足 ZED SDK 官方的 GPU/显存建议。这里采用单帧推理，并让 SDK 与 Depth Net 顺序使用 GPU；实际显存峰值和运行速度尚未在 3070 上测量。建议系统内存至少 16GB，安装期间预留约 15GB 磁盘空间。

**已验证的是 Depth Net + UE 比较。ZED SDK 三方入口属于实验性代码，尚未在 Windows/NVIDIA 上运行成功；包内没有伪造的 SDK 深度。** GPU 驱动、Python、Windows PyTorch、ZED SDK/pyzed 和仿真运行库需要在接收方安装/下载，第一次安装需要联网。本包不是完全离线的 Windows 环境镜像。

## 接收方先运行这三个入口

先完整解压 ZIP，建议放在 `C:\FSAI_3070_Windows_Kit`；不要在压缩包预览窗口里直接运行。

1. 安装 [Python 3.11 x64](https://www.python.org/downloads/windows/)，包含 Python Launcher；更新 NVIDIA 驱动。
2. 双击 `00_setup_windows.cmd`。它只在本文件夹创建 `.venv`，安装 CUDA 12.8 版 PyTorch 与依赖，保存环境检查结果。
3. 双击 `01_verify_and_replay.cmd`，校验文件并重算已有预测的指标。这一步无需 GPU，也不会运行网络。预期：419524 个像素，MAE 约 2.613193 m，RMSE 约 2.747712 m。
4. 双击 `02_run_depthnet.cmd`，在 GPU 0 上重新估计深度。查看 `outputs/mono/comparison.png` 和 `summary.json`。

如需 CPU 排查，在本文件夹的命令提示符运行：

```bat
.venv\Scripts\python.exe scripts\compare.py --device cpu --output outputs\mono_cpu
```

请勿另外 `pip install ultralytics` 来替换源码。这里的模型用到了特定深度头，本包始终优先导入 `vendor/ultralytics` 8.4.189。安装时的依赖实际版本记录在 `installed_requirements.txt`，环境与 GPU 信息记录在 `outputs/environment.json`。

## ZED SDK 三方比较

阅读 `ZED_README.md`，安装 SDK 5.5.x 的 Windows CUDA 12 版本，然后依次运行 `03_install_zed_python.cmd` 和 `04_run_zed_three_way_EXPERIMENTAL.cmd`。若缺少支持仿真流的 DLL，可按说明运行 `03b_download_sim_runtime_IF_NEEDED.cmd`。

实验入口将同一组 UE 左右图反复发送给 SDK，由 SDK 计算深度，再把 SDK 实际接收到的左图交给 Depth Net。它不会把 UE 真值输入 SDK。若运行库导出、版本或相机参数不匹配，程序会停止，不产生误导性的三方指标。SDK 首次运行还可能下载、优化其自己的神经深度模型。

成功后结果在 `outputs/sdk_three_way/comparison/`，包括三方图、SDK/Depth Net 的误差和共同有效像素指标。此结果使用 `stereo_demo` 那一帧，与单目演示不是同一个场景时刻，不能直接拿单目演示的数值做排名。

## 文件内容

| 路径 | 用途 |
|---|---|
| `models/yolo26s-depth.pt` | 当前 FSAI-Seg 使用的深度权重，已随包附带 |
| `vendor/` | 对应 Ultralytics Python 源码、版本信息及许可证 |
| `scripts/compare.py` | 新推理或重放预测；支持额外的同图 SDK 数组；不进行真值拟合缩放 |
| `scripts/prepare_stereo.py` | 双目校正和 UE 光轴 Z 坐标转换 |
| `scripts/run_sdk.py` / `sim_streamer.py` | 实验性 Windows SDK 接收与 RGB 双目流桥接 |
| `data/mono_demo/image_pinhole.png` | 单目测试输入，1280×720，理想针孔图 |
| `data/mono_demo/depth_z_pinhole_m.npy` | 对应针孔图的 UE 光轴 Z，float32 米，HxW |
| `data/mono_demo/image.png` / `depth_z_m.npy` | 带模拟镜头畸变的一组 RGB/Z，不与针孔组混用 |
| `data/mono_demo/instances_pinhole.png` | 与针孔图对齐的 uint16 锥桶实例 ID；0 为背景 |
| `data/mono_demo/annotations*.json` / `cones.csv` | 锥桶 ID、世界/相机坐标、可见范围；坐标以文件注明的参考点为准 |
| `data/mono_demo/capture.json` | 采集姿态、相机、单位与同步约定 |
| `data/stereo_demo/` | 另一时刻的同步左右图、左眼 UE 深度、标定和帧信息；用于 SDK 实验 |
| `reference_results/` | 之前实际跑出的参考预测/图/指标；其中锥桶提取来自旧的场景特定脚本 |
| `validation/` | 本次打包脚本的新 CPU 运行结果和检查记录 |
| `original_fsai_seg/` | 原有三个相关 Python 脚本及 README 的快照，保留作对照 |
| `licenses/` | Ultralytics AGPL、Stereolabs 仿真源码 MIT、第三方声明和 ABI 参考头文件 |
| `MANIFEST_SHA256.json` | 打包文件校验；新增 `.venv` 和输出不参与原始文件校验 |

原始标定文件 `data/stereo_demo/SN2448592.conf` 对应实车原版 ZED；它不等于 SDK 运行后自动校正的 CameraInfo。本次 SDK 实验采用重新校正后的虚拟相机，详细差异在 `ZED_README.md`。

## 怎样读误差

所有比较数组都按“相机光轴向前的 Z，米”解释，不能用归一化灰度深度图代替。模型输出沿用 FSAI-Seg 原脚本的米制 Z 解释；没有用 UE 真值拟合比例或偏移。默认比较 `0 < UE Z <= 20m`，排除 NaN、Inf、非正数。

`summary.json` 的 `pairwise` 是每个方法各自有效的像素，包含覆盖率；`common_pixel_metrics` 是所有方法共同有效的同一批像素，更适合公平对比。SDK 失效的像素会降低覆盖率，不会被当作正确值填进去。

`cones.csv` 按 UE 原生实例 ID 统计可见锥桶表面的逐像素深度误差，不是锥桶底部中心坐标误差。旧 `reference_results/cones_comparison.*` 使用场景特定的地面提取方法，因此像素数略有不同。本次实例掩码统计为 5895 个像素，MAE 约 1.421856 m；不要把两套方法当作完全相同的指标。

图中深度颜色固定在 0–20m，误差颜色固定在 0–5m；超出颜色范围的值只在显示上截断，指标仍使用原始数值。

## 自己提供同图 SDK 输出

已有 SDK 产生的 `MEASURE.DEPTH` 数组时，可以运行：

```bat
.venv\Scripts\python.exe scripts\compare.py --image path\left.png --gt path\ue_z_m.npy --sdk-depth path\sdk_z_m.npy --device 0 --output outputs\custom
```

三个文件必须来自同一帧、相同像素坐标和同一个相机，深度单位必须是米；相同尺寸本身不能证明对齐。脚本不会替你猜测几何关系或自动把错位数组拉伸到一起。

测试：`.venv\Scripts\python.exe -m unittest discover -s tests -v`。结果回传可运行 `05_collect_results.cmd`，发送生成的 `FSAI_results.zip`。

官方资料：[GPU 要求](https://support.stereolabs.com/articles/8942016062-which-graphics-cards-are-supported-by-the-zed-sdk)、[SDK 下载](https://www.stereolabs.com/developers/release)、[Windows 安装](https://docs.stereolabs.com/docs/development/zed-sdk/windows)、[PyTorch 安装](https://pytorch.org/get-started/locally/)。核对日期：2026-10-27。
