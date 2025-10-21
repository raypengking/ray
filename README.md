# Ray Storyboard Pipeline

将中文小说或文章转化为可预览的分镜脚本和对应图像，再导出为 HTML 页面。默认以 Qwen2.5 生成分镜脚本，使用 Nano Banana 渲染分镜图，并通过固定 seed 与 LoRA/Embedding 选项保持人物一致性。

## 功能概览

- **文本拆分与脚本生成**：按自定义长度分段，调用本地或远程大模型（默认 Qwen2.5 OpenAI 兼容接口）输出结构化分镜脚本。
- **Nano Banana 图像渲染**：针对每个镜头的英文提示词调用 Nano Banana 服务，支持 CFG、步数、尺寸、采样器等参数配置。
- **角色一致性策略**：统一基础 seed，支持追加 LoRA 权重与文本 Embedding。
- **HTML 预览导出**：自动生成带有镜头图文信息的沉浸式预览页面及 JSON 数据。
- **占位图模式**：在 `--dry-run` 模式下生成可视化占位图，无需连接外部服务即可预览流程。

## 安装依赖

```bash
pip install -e .[dev]
```

或直接在仓库根目录设置 `PYTHONPATH`：

```bash
export PYTHONPATH=src
pip install -r <(python -c "import tomllib;import sys;d=tomllib.load(open('pyproject.toml','rb'))['project']['dependencies'];print('\n'.join(d))")
```

## 快速开始

1. 准备一个中文文本文件，例如 `input.txt`。
2. 启动本地/远程 Qwen2.5 接口与 Nano Banana 服务。
3. 运行命令：

```bash
python -m ray.cli run input.txt \
  --output outputs/demo \
  --llm-endpoint http://127.0.0.1:8000/v1/chat/completions \
  --nano-endpoint http://127.0.0.1:3921/api/generate \
  --style 电影质感 --style 写实光影
```

执行结束后将在 `outputs/demo` 目录下生成：

- `index.html`：分镜图文预览，可直接在浏览器查看；
- `storyboard.json`：原始分镜结构化数据；
- `images/shot_XXX.png`：每个镜头对应的 Nano Banana 输出或占位图。

## 常用选项

| 选项 | 说明 |
|------|------|
| `--chunk-size` / `--chunk-overlap` | 控制文本拆分长度与重叠，帮助 LLM 处理长文。 |
| `--shots-per-chunk` | 指示每段建议生成的镜头数量。 |
| `--style` | 自定义风格关键词，可多次传入。 |
| `--keep-dialogue/--no-keep-dialogue` | 是否严格保留原文对白。 |
| `--lora 路径[:权重]` | 为 Nano Banana 请求附加 LoRA。 |
| `--embedding 路径` | 加载额外的 Embedding。 |
| `--dry-run` | 启用占位图和 Mock LLM，便于离线调试。 |

所有命令行参数均可通过 `python -m ray.cli run --help` 查看。

## 环境变量

- `QWEN_API_KEY`：若 Qwen 接口需要鉴权，可在环境变量中配置。

## 工作流程

1. **拆分文本**：`ray.text_processing.split_text` 按标点与长度切分，保留适度重叠。
2. **生成脚本**：`ray.storyboard_generator.StoryboardGenerator` 构建 JSON 格式提示词调用 LLM，并解析返回的镜头结构。
3. **渲染图像**：`ray.image_generation.NanoBananaClient` 为每个镜头提交请求，若失败或 dry-run 则生成带提示信息的占位图。
4. **导出预览**：`ray.html_exporter.StoryboardHTMLExporter` 渲染模板输出 HTML 与 JSON。

## 扩展与自定义

- 可通过继承 `LLMClient` 实现自定义模型调用逻辑。
- 替换模板目录以适配不同的品牌视觉或排版风格。
- 调整 `NanoBananaConfig` 以适配不同显卡、模型、采样策略。

## 许可

MIT License。
