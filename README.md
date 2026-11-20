# Photo Pose

手机端拍照姿势 MVP。实时取景、场景识别和人体骨架在浏览器运行。开启本地 ShutterMuse 后，拍下或选中的照片会交给同一台电脑上的模型分析。内置姿势为示意图，真实照片需要完成授权审核。

## 启动

需要 Node.js 20.19+ 或 22.12+。执行 `npm install`、`npm run dev`；电脑本机可用 localhost。手机需访问可信 HTTPS 地址（普通局域网 HTTP 不能使用相机），可用自己的 HTTPS 开发代理。运行 `npm test`、`npm run build` 检查。

MediaPipe WASM 和姿态模型随项目提供，首次加载需联网获取 TensorFlow MobileNet 模型；国内网络可能受限。模型文件和 npm 依赖的许可、部署可用性在正式发布前需单独核验。场景是基于通用分类标签的粗略建议，并非专用场景模型；识别不确定时请手动选择。姿势纠正是二维近似，需让全身入镜并保持拍摄角度接近参考。

## 素材准备

查阅 `docs/素材候选.md`；候选链接不代表已获展示或本地解析授权。获得照片权利人及必要肖像许可并确认允许本地 Codex 解析后，由本地 Codex 查看图片，记录场景、动作、景别和描述；打开 `/material.html` 使用本地姿态模型提取 33 个归一化关键点并人工检查，下载 JSON。将图片与 JSON 一起导入页面。JSON 格式示例见 `tools/pose-template.json`。导入只保存在当前页面会话，刷新后需重新导入；避免上传未授权素材。



## 上传照片分析

在拍摄页选择「上传照片」，可从设备选择图片，在本机进行场景建议和人体骨架分析；也可以手动修改场景并选择参考姿势。模型资源首次加载可能需要网络。普通开发模式在浏览器会话中处理照片；本地 ShutterMuse 模式会自动将照片缩小后发送到本机推理服务。选择姿势素材的「导入素材」仍需要单独的图片及关键点 JSON。

## 本地 ShutterMuse 姿势与构图建议

在项目根目录执行 `npm ci` 安装网页依赖，然后选择一个后端。两个命令都启动同一个[本机试拍页](http://127.0.0.1:4174/shuttermuse.html)。模型服务监听 `127.0.0.1:8081`，推理时照片只发送给本机。按 Ctrl+C 同时关闭网页和模型服务。

| 后端 | 第一次启动 | 项目命令 |
| --- | --- | --- |
| llama.cpp | 需要 Python 3 和 `llama-server`（macOS 可用 `brew install llama.cpp`）。下载 ShutterMuse Q4_K_M 和配套视觉投影，共约 6.19 GB。 | `npm run shuttermuse -- llama` |
| MLX | 需要 Apple Silicon 和 `uv`（`brew install uv`）。在独立的 Python 3.12 环境中下载官方 BF16 权重（约 17.6 GB），并转换为 4-bit。磁盘需能同时容纳原始权重、转换结果和依赖。 | `npm run shuttermuse -- mlx` |

模型按后端分别缓存在 `.pose-models/`，也可以通过 `SHUTTERMUSE_CACHE_DIR` 指定目录。重启会复用已有权重。模型就绪后，点「去相机拍照」按下快门，或在试拍页选一张照片。原图旁会自动显示一句可选的姿势建议。所有照片使用同一条后台指令。如果当前姿势已经合适，模型可以建议保持。

需要对照骨架或裁切范围时，展开「姿势与构图工具」。点「查看姿势骨架」可查看建议关节，点「查看构图建议」可查看取景建议。点「查看原片」可回到照片和最初的建议。[官方构图示例照片](https://github.com/lijayuTnT/ShutterMuse/blob/main/test/401128801616615964.webp)

单独调用模型获取一句拍摄建议，可运行 `node tools/shuttermuse-infer.mjs public/pose-images/mountain-sunrise.jpg advice`。单独启动模型的命令是 `python3 tools/shuttermuse-server.py llama` 或 `python3 tools/shuttermuse-server.py mlx`。[官方任务说明](https://github.com/lijayuTnT/ShutterMuse#quick-start) · [官方权重](https://huggingface.co/ShutterMuse/ShutterMuse) · [社区 GGUF 权重](https://huggingface.co/prithivMLmods/ShutterMuse-GGUF)

### 山顶背影实测

把项目自带的山顶照片缩至 640 × 427，用 llama.cpp 在 CPU 上运行 ShutterMuse Q4_K_M，单次耗时约 117 秒。模型建议面向山峦、双臂展开、左腿迈步。绿色骨架画出了双臂展开，但双膝和脚踝几乎等高，图上很难看出迈步。打开[原始照片](public/pose-images/mountain-sunrise.jpg)、[建议骨架叠加图](docs/shuttermuse/sample-overlay.jpg)和[本次返回的 17 个坐标与文字](docs/shuttermuse/sample-pose.json)可对照查看。
