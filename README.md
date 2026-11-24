# Photo Pose

手机端拍照姿势 MVP。实时取景、场景识别和人体骨架在浏览器运行。开启本地 ShutterMuse 后，拍下或选中的照片会交给同一台电脑上的模型分析。内置姿势为示意图，真实照片需要完成授权审核。

## 启动

需要 Node.js 20.19+ 或 22.12+。执行 `npm install`、`npm run dev`；电脑本机可用 localhost。手机需访问可信 HTTPS 地址（普通局域网 HTTP 不能使用相机），可用自己的 HTTPS 开发代理。运行 `npm test`、`npm run build` 检查。

MediaPipe WASM 和姿态模型随项目提供，首次加载需联网获取 TensorFlow MobileNet 模型；国内网络可能受限。模型文件和 npm 依赖的许可、部署可用性在正式发布前需单独核验。场景是基于通用分类标签的粗略建议，并非专用场景模型；识别不确定时请手动选择。姿势纠正是二维近似，需让全身入镜并保持拍摄角度接近参考。

## 素材准备

查阅 `docs/素材候选.md`；候选链接不代表已获展示或本地解析授权。获得照片权利人及必要肖像许可并确认允许本地 Codex 解析后，由本地 Codex 查看图片，记录场景、动作、景别和描述；打开 `/material.html` 使用本地姿态模型提取 33 个归一化关键点并人工检查，下载 JSON。将图片与 JSON 一起导入页面。JSON 格式示例见 `tools/pose-template.json`。导入只保存在当前页面会话，刷新后需重新导入；避免上传未授权素材。

## 本地场景向量检索

上传一张现场照片，可以在参考图库里查找背景相似的照片，参考人物的站位和动作。DINOv2-Small 将场景编码为 384 维向量，存入本机 LanceDB。查询时用同一模型编码照片，检测到人物时先遮住人物，再按余弦相似度排列参考照片。分数越高，表示背景越相似。工具再按机位、景别和可见关节筛选姿势，将入选参考照的骨架叠加到查询照片上。

在项目根目录用 `uv` 依次运行以下命令。第一条下载 CaptureGuide-Bench 研究样本并提取关键点，第二条用样本重建向量表，对自带的长椅和小巷照片执行检索，第三条启动本机上传页面。首次运行需要联网获取样本、依赖和模型权重。

```sh
uv run --python 3.12 --with 'mediapipe==0.10.21' --with 'numpy<2' --with pillow tools/pose_retrieval_samples.py
uv run tools/pose_retrieval.py
uv run tools/pose_retrieval_web.py --output-dir .pose-local/retrieval
```

打开[本机检索页](http://127.0.0.1:4186/)查看示例或上传自己的照片。命令行示例的得分和骨架坐标保存在 `.pose-local/retrieval/report/matches.json`。未标注的照片可由本机 ShutterMuse 估计机位，也可以在上传时手动选择查询照片的机位。

## 上传照片分析

在拍摄页选择「上传照片」，可从设备选择图片，在本机进行场景建议和人体骨架分析；也可以手动修改场景并选择参考姿势。模型资源首次加载可能需要网络。普通开发模式在浏览器会话中处理照片；本地 ShutterMuse 模式会自动将照片缩小后发送到本机推理服务。选择姿势素材的「导入素材」仍需要单独的图片及关键点 JSON。

## 本地 ShutterMuse 姿势与构图建议

在项目根目录执行 `npm ci` 安装网页依赖，然后选择一个后端。两个命令都启动同一个[本机试拍页](http://127.0.0.1:4174/shuttermuse.html)。模型服务监听 `127.0.0.1:8081`，推理时照片只发送给本机。按 Ctrl+C 同时关闭网页和模型服务。

| 后端 | 第一次启动 | 项目命令 |
| --- | --- | --- |
| llama.cpp | 需要 Python 3 和 `llama-server`（macOS 可用 `brew install llama.cpp`）。下载 ShutterMuse Q4_K_M 和配套视觉投影，共约 6.19 GB。 | `npm run shuttermuse -- llama` |
| MLX | 需要 Apple Silicon 和 `uv`（`brew install uv`）。在独立的 Python 3.12 环境中下载官方 BF16 权重（约 17.6 GB），并转换为 4-bit。磁盘需能同时容纳原始权重、转换结果和依赖。 | `npm run shuttermuse -- mlx` |

模型按后端分别缓存在 `.pose-models/`，也可以通过 `SHUTTERMUSE_CACHE_DIR` 指定目录。重启会复用已有权重。模型就绪后，点「去相机拍照」按下快门，或在试拍页选一张照片。三个姿势任务分两次调用本机模型。第一步读取原片，结合现场和当前景别写出具体动作。第二步读取同一张原片和刚生成的动作，预测目标姿势的 17 个关节及可见性。点「换个姿势」时，模型还会收到上一条建议。

### 试跑论文图中的五类任务

试拍页另有五张独立摄影原片。选择「摄影实测」里的照片会重新调用本机模型，也可以直接查看首轮的原片、骨架、模型原话和逐图核对记录。[五张摄影实测对照](public/field-cases/index.html)

在试拍页的「论文示例原片」选择照片，点「试用示例」即可运行对应任务；也可以先选「试拍任务」，再上传自己的照片。九个示例取自用户提供的 ShutterMuse 展示图，部分画面保留了作者的网格或箭头。本机对照图使用示例 PNG 直接推理；交互试拍会把照片转成 JPEG 后推理，结果可能略有变化。页面可分别查看原片、模型预测和实际裁切。[作者项目](https://github.com/lijayuTnT/ShutterMuse) · [论文展示图](public/pose-examples/paper-figure.jpg) · [九张本机实测](public/pose-examples/local-results.jpg)

| 任务 | 示例 | 可以对照的结果 |
| --- | --- | --- |
| a · 构图判断 | 街巷人像、倾斜湖面、光影人像 | 模型选择保留、裁切或重新取景。需要退后重拍时，以文字提示摄影师。 |
| b · 指令裁切 | 湖边人像、山景人像 | 分别试横向 4:3 和半身。切换「按建议裁切」查看保留下来的原片像素。 |
| c · 空场景姿势 | 台阶、江边护栏 | 从空景预测人物站位、动作和关节，生成示意图时在场景里加入人物。 |
| d · 构图与姿势 | 果园 | 在原片上同时显示建议画框和目标骨架。生图时裁切场景，并把关节换算到裁后坐标。 |
| e · 人像姿势调整 | 咖啡馆 | 根据已有的人物推荐动作，可切换原片与骨架，或点「换个姿势」。 |

三个姿势任务支持填写拍摄要求。试拍页请求最多三种动作，先显示第一种，再为其他候选逐个计算骨架。点选就绪方案可以对照文字和关节，再点「生成 AI 示意图」生成这一种动作。模型分析、框选和照片裁切均可单独在本机完成。论文作者使用 GPT-Image-2 将关键点渲染成人像。当前项目使用本机网关配置的图像编辑渠道。[作者演示说明](https://lijayutnt.github.io/ShutterMuse/)

要在模型启动后重跑果园原图，执行 `node tools/shuttermuse-infer.mjs public/pose-examples/joint-orchard.png joint`；4:3 示例可运行 `node tools/shuttermuse-infer.mjs public/pose-examples/crop-ratio.png crop 4:3 368 275`。

单独调用模型获取一句拍摄建议，可运行 `node tools/shuttermuse-infer.mjs public/pose-images/mountain-sunrise.jpg advice`。单独启动模型的命令是 `python3 tools/shuttermuse-server.py llama` 或 `python3 tools/shuttermuse-server.py mlx`。[官方任务说明](https://github.com/lijayuTnT/ShutterMuse#quick-start) · [官方权重](https://huggingface.co/ShutterMuse/ShutterMuse) · [社区 GGUF 权重](https://huggingface.co/prithivMLmods/ShutterMuse-GGUF)

### 山顶背影实测

把项目自带的山顶照片缩至 640 × 427，用 llama.cpp 在 CPU 上运行 ShutterMuse Q4_K_M，单次耗时约 117 秒。模型建议面向山峦、双臂展开、左腿迈步。绿色骨架画出了双臂展开，但双膝和脚踝几乎等高，图上很难看出迈步。打开[原始照片](public/pose-images/mountain-sunrise.jpg)、[建议骨架叠加图](docs/shuttermuse/sample-overlay.jpg)和[本次返回的 17 个坐标与文字](docs/shuttermuse/sample-pose.json)可对照查看。

### 按拍摄建议生成 AI 示意图

在本机 `agent-gateway` 仓库运行 `HTTP_ADDR=127.0.0.1:8080 go run ./cmd/gateway`。如果 ShutterMuse 已在 8081 运行，在 Photo Pose 目录执行 `npm run shuttermuse -- existing`。首次启动模型仍用上面的 `llama` 或 `mlx` 命令。若网关用了其他端口，在 Photo Pose 进程设置 `AGENT_GATEWAY_PORT`。

拍照或选图后先在本机生成一句建议。点「生成 AI 示意图」后，Photo Pose 通过网关的游客登录获取会话，用干净照片（长边最多 1280 像素）、当前方案的独立骨架参考图、可见关节的归一化坐标和动作指令调用 `POST /v1/images/edits`，再按返回的任务编号查询结果。请求使用 `model: "group:image_edit"`，可以在 Portal 的 Route Groups → `image_edit` 中切换默认图像编辑模型。图像渠道和供应商密钥由网关管理。每次安装的游客设备 ID 保存在 `.pose-local/gateway-device-id`。如需使用已有网关账号，可在服务端设置 `AGENT_GATEWAY_TOKEN` 使用该账号的有效访问令牌。

两张输入图按原片、目标骨架的顺序提交给支持多图输入的编辑模型。网关会记录提交的照片和指令，并通过图像编辑渠道返回成片的 HTTPS 链接。生成可能产生费用；取消等待后，网关已受理的任务仍可继续。
