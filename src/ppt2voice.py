#!/usr/bin/env python3
import os
import sys
import asyncio
from pathlib import Path
from typing import List, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import edge_tts
from tqdm import tqdm
import requests
import urllib3
import ssl
import base64
from PyPDF2 import PdfReader, PdfWriter
import time
import aiofiles

# 配置SSL
ssl._create_default_https_context = ssl._create_unverified_context

# 配置urllib3
urllib3.disable_warnings()

class PPT2Voice:
    def __init__(self):
        load_dotenv()
        self.setup_environment()
        self.setup_gemini()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def setup_environment(self):
        """设置环境变量，包括代理"""
        try:
            # 设置代理
            http_proxy = os.getenv("HTTP_PROXY")
            https_proxy = os.getenv("HTTPS_PROXY")
            
            if http_proxy and https_proxy:
                # 设置环境变量
                os.environ["HTTPS_PROXY"] = https_proxy
                os.environ["HTTP_PROXY"] = http_proxy
                
                # 禁用SSL验证
                os.environ["REQUESTS_CA_BUNDLE"] = ""
                os.environ["SSL_CERT_FILE"] = ""
                
                print(f"已设置代理: {https_proxy}")
        except Exception as e:
            print(f"警告：环境设置出错: {str(e)}")

    def setup_gemini(self):
        """设置Gemini API"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("请在.env文件中设置GOOGLE_API_KEY")
        
        # 从环境变量读取模型名称，如果未设置则使用默认值
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")
        print(f"使用模型: {model_name}")
        
        # 创建自定义的 requests session
        session = requests.Session()
        session.verify = False
        
        if os.getenv("HTTPS_PROXY"):
            session.proxies = {
                "http": os.getenv("HTTP_PROXY"),
                "https": os.getenv("HTTPS_PROXY")
            }
            
        # 配置 Google API
        genai.configure(
            api_key=api_key,
            transport="rest"
        )
        
        # 配置安全设置
        safety_settings = {
            "harassment": "block_none",
            "hate_speech": "block_none",
            "sexually_explicit": "block_none",
            "dangerous": "block_none",
        }
        
        max_retries = 10  # 最大重试次数
        base_delay = 60  # 基础延迟时间（秒）
        
        for attempt in range(max_retries):
            try:
                self.model = genai.GenerativeModel(
                    model_name=model_name,
                    safety_settings=safety_settings
                )
                
                # 测试连接
                test_prompt = "Hello"
                response = self.model.generate_content(test_prompt)
                print("API连接测试成功")
                return  # 成功后直接返回
                
            except Exception as e:
                if "429" in str(e):  # 配额限制错误
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (2 ** attempt)  # 指数退避
                        print(f"初始化时遇到配额限制，等待 {wait_time} 秒后重试... (第 {attempt + 1}/{max_retries} 次)")
                        time.sleep(wait_time)  # 使用同步sleep，因为这是在同步函数中
                    else:
                        print(f"警告：无法设置安全配置或测试连接失败：{str(e)}")
                        # 在最后一次尝试时，使用基本配置
                        self.model = genai.GenerativeModel(model_name=model_name)
                else:
                    print(f"警告：设置时遇到非配额错误：{str(e)}")
                    self.model = genai.GenerativeModel(model_name=model_name)
                    break

    async def generate_lecture_content(self, pdf_path: str) -> str:
        """使用Gemini生成完整的讲解内容"""
        try:
            # 读取PDF文件
            pdf = PdfReader(pdf_path)
            total_pages = len(pdf.pages)
            print(f"PDF共有 {total_pages} 页")
            
            # 计算需要分成几批
            batch_size = 10
            num_batches = (total_pages + batch_size - 1) // batch_size
            
            all_responses = []
            
            # 处理每一批
            for batch_idx in range(num_batches):
                start_page = batch_idx * batch_size
                end_page = min((batch_idx + 1) * batch_size, total_pages)
                
                print(f"\n处理第 {start_page + 1} 到 {end_page} 页...")
                
                # 创建临时PDF文件包含当前批次的页面
                temp_pdf = PdfWriter()
                for page_num in range(start_page, end_page):
                    temp_pdf.add_page(pdf.pages[page_num])
                
                temp_path = f'temp_batch_{batch_idx}.pdf'
                with open(temp_path, 'wb') as f:
                    temp_pdf.write(f)
                
                # 读取临时PDF文件
                with open(temp_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                # 删除临时文件
                os.remove(temp_path)
                
                # 创建PDF部分
                pdf_part = {
                    "mime_type": "application/pdf",
                    "data": base64.b64encode(pdf_bytes).decode('utf-8')
                }
                
                # 构建提示
                batch_prompt = f"""你是一位资深的数学物理教授，正在为研究生讲解一份数学物理相关的幻灯片。
这是PDF文档的第{start_page + 1}页到第{end_page}页，共{total_pages}页。请按照以下要求给出这些页面的讲解：

1. 讲解风格要求：
   - 语言自然流畅，像真人授课
   - 专业严谨，数学物理概念准确
   - 体现出深厚的学术功底
   - 讲解有层次感，由浅入深
   - 讲解要详细，不要有遗漏，对重要的概念和知识还要做适当的拓展和延伸
   - 善于类比和举例，帮助理解
   - 不要有类似于### ，**， 这样的格式，因为会影响语音的朗读
   - 不要总是提到"这张幻灯片"这类的句子，就好像在念幻灯片上的内容一样
   - 不要有念幻灯片的感觉
   - 要像一个老师在讲课，而不是一个念稿子的人 
   

2. 内容要求：
   - 将所有数学公式转换为口语化的专业表达
   - 原则就是让学生通过语音就可以准确理解公式的内容
   - 例如：将 "∇×E = -∂B/∂t" 表达为 "旋度E等于负的B对时间的偏导"
   - 例如：将 "∫f(x)dx" 表达为 "对f(x)关于x积分"
   - 例如：将 "dx²/dt" 表达为 "x对时间的二阶导数"
   - 说明公式的物理意义和应用场景
   - 指出关键概念之间的联系
   - 适当补充理论背景和发展历史
   - 分享你的学术见解和研究经验

3. 结构要求：
   - 每页内容前加入"第X页讲解"的标题
   - 使用txt格式输出
   - 所有数学公式必须用口语化的方式表达，不要使用LaTeX格式
   - 适当使用小标题和列表增加可读性
   - 确保所有内容都是以口语化的方式表达，便于朗读理解

4. 数学符号的口语化规则：
   - 上标：表达为"的n次方"，例如 "x²" 读作 "x的平方"
   - 下标：表达为"下标"，例如 "aₙ" 读作 "a下标n"
   - 分数：使用"分之"，例如 "1/2" 读作 "二分之一"
   - 希腊字母：直接读出名称，例如 "α" 读作 "阿尔法"，"β" 读作 "贝塔"
   - 积分：明确说明积分区间和变量，例如 "从0到1对x积分"
   - 导数：明确说明是对什么求导，例如 "对时间求导"
   - 偏导：说明"偏导"而不是"导数"
   - symbol ∇：读Del, 而不是nabla
   - 矩阵：说明行列数和元素位置，例如 "2乘2的矩阵"
   - 向量：说明是"向量"，例如 "向量a"
   - 特殊函数：完整读出函数名称，例如 "正弦函数"而不是"sin"

请确保按顺序处理每一页，每页都要以"第X页讲解"开始。所有数学公式和概念都要用自然的口语表达，让听众能够清晰理解。"""

                max_retries = 10  # 恢复到10次重试
                base_delay = 30  # 恢复到30秒基础延迟
                
                for attempt in range(max_retries):
                    try:
                        # 创建新的聊天会话
                        chat = self.model.start_chat(history=[])
                        
                        # 发送PDF和提示
                        response = chat.send_message(
                            [batch_prompt, pdf_part],
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.7,
                                top_p=0.8,
                                top_k=40,
                                max_output_tokens=8192,
                                candidate_count=1
                            ),
                            stream=False
                        )
                        
                        all_responses.append(response.text)
                        print(f"第 {start_page + 1} 到 {end_page} 页处理完成")
                        break
                        
                    except Exception as e:
                        if "429" in str(e):  # 配额限制错误
                            if attempt < max_retries - 1:
                                wait_time = base_delay * (2 ** attempt)  # 指数退避
                                print(f"遇到配额限制，等待 {wait_time} 秒后重试... (第 {attempt + 1}/{max_retries} 次重试)")
                                await asyncio.sleep(wait_time)
                            else:
                                raise Exception(f"处理第 {start_page + 1} 到 {end_page} 页时遇到配额限制，已达到最大重试次数")
                        else:
                            raise e
                
                # 在批次之间添加延迟，避免触发限制
                if batch_idx < num_batches - 1:
                    wait_time = 15  # 恢复到15秒
                    print(f"等待 {wait_time} 秒后处理下一批...")
                    await asyncio.sleep(wait_time)
            
            # 合并所有响应
            full_content = "\n\n".join(all_responses)
            
            # 保存完整的markdown文件
            base_name = Path(pdf_path).stem
            markdown_file = self.output_dir / f"{base_name}_lecture.md"
            markdown_file.write_text(full_content, encoding="utf-8")
            
            return full_content
            
        except Exception as e:
            raise Exception(f"生成内容失败: {str(e)}")

    async def text_to_speech(self, text: str, output_file: str):
        """将文本转换为语音"""
        try:
            print("\n开始生成音频...")
            # 计算大约的字数和预计时间
            total_chars = len(text)
            # 假设每分钟朗读180个字
            estimated_minutes = total_chars / 180
            estimated_seconds = estimated_minutes * 60
            print(f"文本总计 {total_chars} 字")
            print(f"预计需要 {int(estimated_minutes)} 分 {int(estimated_seconds % 60)} 秒")
            
            # 创建进度条
            progress_bar = tqdm(
                total=total_chars,
                desc="生成音频",
                unit='字',
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}字 [{elapsed}<{remaining}]'
            )
            
            # 创建通信对象
            communicate = edge_tts.Communicate(text, "zh-CN-YunxiNeural")
            
            # 记录已处理的字符数
            processed_chars = 0
            
            # 使用异步写入方式
            async with aiofiles.open(output_file, mode="wb") as file:
                async for chunk in communicate.stream():
                    try:
                        if chunk["type"] == "audio":
                            await file.write(chunk["data"])
                            # 更新进度条（基于音频数据块）
                            processed_chars = min(total_chars, processed_chars + len(chunk["data"]) // 100)
                            progress_bar.n = processed_chars
                            progress_bar.refresh()
                        elif chunk["type"] == "WordBoundary":
                            # 如果有可用的文本偏移信息，使用它更新进度
                            if "text" in chunk:
                                text_length = len(chunk.get("text", ""))
                                processed_chars = min(total_chars, processed_chars + text_length)
                                progress_bar.n = processed_chars
                                progress_bar.refresh()
                    except Exception as chunk_error:
                        print(f"处理数据块时出现警告（继续处理）: {str(chunk_error)}")
                        continue
            
            # 完成进度条
            progress_bar.n = total_chars
            progress_bar.refresh()
            progress_bar.close()
            
            # 获取生成的音频文件大小
            audio_size = os.path.getsize(output_file) / (1024 * 1024)  # 转换为MB
            print(f"\n音频文件生成完成！文件大小：{audio_size:.1f}MB")
            
        except Exception as e:
            print(f"转换语音时出错：{str(e)}")
            raise e

    def split_content_by_pages(self, content: str) -> List[Tuple[int, str]]:
        """将生成的内容按页拆分"""
        pages = []
        current_page = []
        current_page_num = None
        
        for line in content.split('\n'):
            if line.startswith('## 第') and '页讲解' in line:
                if current_page_num is not None:
                    pages.append((current_page_num, '\n'.join(current_page)))
                    current_page = []
                try:
                    current_page_num = int(line[3:line.index('页')])
                except ValueError:
                    continue
            current_page.append(line)
            
        if current_page_num is not None:
            pages.append((current_page_num, '\n'.join(current_page)))
            
        return pages

    async def process_pdf(self, pdf_path: str):
        """处理PDF文件"""
        if not os.path.exists(pdf_path):
            print(f"错误：找不到文件 {pdf_path}")
            return

        print("正在生成讲解内容...")
        try:
            lecture_content = await self.generate_lecture_content(pdf_path)
        except Exception as e:
            print(f"生成讲解内容时出错：{str(e)}")
            return
        
        # 生成单个音频文件
        print("正在生成音频文件...")
        try:
            audio_file = self.output_dir / f"{Path(pdf_path).stem}_lecture.mp3"
            await self.text_to_speech(lecture_content, str(audio_file))
            print(f"\n处理完成！")
            print(f"讲解文稿已保存至：{self.output_dir / f'{Path(pdf_path).stem}_lecture.md'}")
            print(f"音频文件已保存至：{audio_file}")
        except Exception as e:
            print(f"生成音频文件时出错：{str(e)}")
            return

async def main():
    if len(sys.argv) != 2:
        print("使用方法: python ppt2voice.py <pdf_file>")
        sys.exit(1)

    converter = PPT2Voice()
    await converter.process_pdf(sys.argv[1])

if __name__ == "__main__":
    asyncio.run(main()) 