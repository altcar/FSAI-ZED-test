# 单次深度比较

使用 FSAI-Seg 原有脚本和 yolo26s-depth.pt 新推理一次，没有修改仓库文件。
输入是仓库的 image_pinhole.png 和 depth_z_pinhole_m.npy，SHA256 核对与 UE shot 000005 完全一致。
网络输出恢复为原图 720×1280；没有额外的相机校正，也没有利用真值拟合预测尺度。

- single_depth_check.png：20 米内的深度、误差和误差区域图。
- single_depth_check.prediction.npy：本次新推理的原始浮点深度。
- single_depth_check.error.npy：有效像素绝对误差，其他像素为 NaN。
- single_depth_check.top_pixels.csv：误差最大的 100 个像素。
- cones_comparison.png / .csv / .json：按现有脚本提取的 11 个锥桶区域的逐个比较。
- summary.json：输入来源、模型指纹、指标和缺失的 ZED SDK 数据说明。

当前完成的是 Depth Net vs UE 真值，并非 ZED SDK / Depth Net / UE 三方比较。
FSAI-Seg 仓库没有 ZED SDK 深度生成入口；还需要同场景对应左图的 SDK 浮点深度，才能增加第三方。
可沿用当前简化方式：核对对应图片、数组尺寸和米制单位，先不增加严格相机标定流程。

现有 cone_depth_comparison.py 的地面拟合、下方行号和绘图裁切针对这张图，不是通用锥桶分割；
新 shot 若有 instances_pinhole.png，可以直接用真实实例掩码，省去地面拟合。
