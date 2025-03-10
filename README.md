# PPT2Voice

一个将 PDF 演示文稿转换为语音讲解的工具。使用 Google Gemini API 生成讲解文稿，并使用 Edge TTS 将文字转换为语音。

## 功能特点

- 支持 PDF 文件输入
- 使用 Google Gemini API 生成专业的讲解内容
- 自动将数学公式转换为口语化表达
- 使用 Edge TTS 生成自然的语音输出
- 支持批量处理大型 PDF 文件
- 实时显示处理进度

## 安装要求

1. Python 3.8 或更高版本
2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 环境配置

1. 复制 `.env.example` 为 `.env`
2. 在 `.env` 文件中设置以下环境变量：
   - `GOOGLE_API_KEY`：Google Gemini API 密钥
   - `HTTP_PROXY`：（可选）HTTP 代理
   - `HTTPS_PROXY`：（可选）HTTPS 代理

## 使用方法

```bash
python src/ppt2voice.py <pdf_file>
```

例如：
```bash
python src/ppt2voice.py test.pdf
```

## 输出文件

程序会在 `output` 目录下生成两个文件：
1. `{pdf_name}_lecture.md`：生成的讲解文稿
2. `{pdf_name}_lecture.mp3`：生成的语音文件

## 注意事项

- 确保有足够的磁盘空间存储生成的音频文件
- 处理大型 PDF 文件时可能需要较长时间
- 建议使用稳定的网络连接 