# LLM.py - 修复版（调整界面布局）
import json
import numpy as np
from typing import List, Dict, Optional
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from database import get_vector_database
import re
import os

# 设置使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class ImprovedMedicalAssistant:
    def __init__(self,
                 llm_model="Qwen/Qwen2.5-1.5B-Instruct",
                 use_gpu=False):

        print(f"初始化医疗助手，使用模型: {llm_model}")

        # 加载向量数据库
        self.vector_db = get_vector_database()
        if self.vector_db is None:
            print("警告: 向量数据库加载失败，将使用回退模式")

        # 加载LLM
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"使用设备: {self.device}")

        try:
            # 加载tokenizer和model
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_model,
                trust_remote_code=True,
                padding_side="left"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            # 创建pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )

            self.llm_loaded = True
            print("LLM加载成功")

        except Exception as e:
            print(f"LLM加载失败: {e}")
            print("将使用基于规则的回复")
            self.llm_loaded = False

        # 对话历史
        self.conversation_history = []
        self.max_history = 5

    def retrieve_relevant_info(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索相关信息 - 修复版：从文本中提取问题和答案
        """
        if self.vector_db is None:
            return []

        try:
            results = self.vector_db.search(query, top_k=top_k, threshold=0.5)

            # 调试信息
            print(f"\n[检索] 查询: '{query}'")
            print(f"[检索] 找到 {len(results)} 个结果")

            # 过滤和提取信息
            filtered_results = []
            for i, result in enumerate(results):
                if result['score'] > 0.5:
                    # 尝试从不同位置提取问题和答案
                    question = ""
                    answer = ""

                    # 方法1: 直接从result中获取
                    if 'question' in result and result['question']:
                        question = result['question']
                    if 'answer' in result and result['answer']:
                        answer = result['answer']

                    # 方法2: 从metadata中获取
                    if (not question or not answer) and 'metadata' in result:
                        metadata = result['metadata']
                        if 'question' in metadata and metadata['question']:
                            question = metadata['question']
                        if 'answer' in metadata and metadata['answer']:
                            answer = metadata['answer']

                    # 方法3: 从text字段解析
                    if (not question or not answer) and 'text' in result:
                        text = result['text']
                        # 尝试解析 "问题：xxx 答案：xxx" 格式
                        if "问题：" in text and "答案：" in text:
                            parts = text.split("答案：", 1)
                            if len(parts) > 1:
                                question_part = parts[0].replace("问题：", "").strip()
                                answer = parts[1].strip()
                                if not question:
                                    question = question_part

                    # 方法4: 从chunk中获取
                    if (not question or not answer) and 'chunk' in result:
                        chunk = result['chunk']
                        if 'text' in chunk:
                            text = chunk['text']
                            if "问题：" in text and "答案：" in text:
                                parts = text.split("答案：", 1)
                                if len(parts) > 1:
                                    question_part = parts[0].replace("问题：", "").strip()
                                    answer = parts[1].strip()
                                    if not question:
                                        question = question_part

                        if 'content' in chunk and not answer:
                            answer = chunk['content']

                    # 如果仍然没有答案，使用text的前100个字符
                    if not answer and 'text' in result:
                        answer = result['text'][:100]

                    # 如果仍然没有问题，使用查询或留空
                    if not question:
                        question = query[:50]

                    # 清理答案
                    if answer:
                        # 移除可能的"答案："前缀
                        if answer.startswith("答案："):
                            answer = answer[3:].strip()

                        # 限制长度
                        if len(answer) > 200:
                            answer = answer[:200] + "..."

                    print(f"[结果{i + 1}] 分数: {result['score']:.3f}, 问题: '{question[:30]}...', 答案: '{answer[:30]}...'")

                    filtered_results.append({
                        'question': question,
                        'answer': answer,
                        'department': result.get('department', ''),
                        'score': result['score']
                    })

            print(f"[过滤] 最终保留 {len(filtered_results)} 个结果")
            return filtered_results

        except Exception as e:
            print(f"检索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def create_prompt(self, query: str, context: List[Dict], history: List[str]) -> str:
        """
        创建提示词 - 简化版
        """
        # 系统提示
        system_prompt = """你是一个专业的医疗助手，请用中文简洁回答用户的问题。
如果以下信息有用，请基于信息回答，否则根据你的知识回答。
最后提醒用户咨询专业医生。"""

        # 添加上下文
        context_text = ""
        if context:
            context_text = "\n\n参考信息："
            for i, ctx in enumerate(context, 1):
                context_text += f"\n{i}. {ctx['answer']}"

        # 完整提示
        full_prompt = f"""{system_prompt}{context_text}

用户问题：{query}

请直接给出答案："""

        return full_prompt

    # 在 generate_answer_with_llm 方法中修改
    def generate_answer_with_llm(self, prompt: str) -> str:
        """使用LLM生成回答 - 修复重复问题"""
        if not self.llm_loaded:
            return "系统正在维护中，请稍后再试。"

        try:
            # 添加生成参数，防止重复
            generation_config = {
                "max_new_tokens": 512,
                "num_return_sequences": 1,
                "pad_token_id": self.tokenizer.eos_token_id,
                "truncation": True,
                "temperature": 0.3,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.2,  # 增加重复惩罚
                "no_repeat_ngram_size": 3,  # 防止3-gram重复
            }

            outputs = self.pipe(
                prompt,
                **generation_config
            )

            generated_text = outputs[0]['generated_text']

            # 提取回答部分（去掉prompt）
            answer = generated_text[len(prompt):].strip()

            # 强化清理逻辑
            answer = self.clean_answer(answer)

            return answer

        except Exception as e:
            print(f"LLM生成失败: {e}")
            return "抱歉，生成回答时出现错误。"

    # 修改 clean_answer 方法
    def clean_answer(self, text: str) -> str:
        """清理回答文本 - 修复重复问题"""
        if not text:
            return "未找到相关信息。"

        # 1. 移除重复的句子或段落
        import re
        sentences = re.split(r'[。！？；\n]+', text)
        unique_sentences = []
        seen_sentences = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 检查句子是否相似（简单去重）
            simplified = sentence.replace(" ", "").replace("，", "").replace("。", "")
            if simplified not in seen_sentences and len(sentence) > 3:
                seen_sentences.add(simplified)
                unique_sentences.append(sentence)

        # 2. 限制总句子数
        if len(unique_sentences) > 8:
            # 保留重要部分，移除后面的重复部分
            important_keywords = ["药物", "用药", "剂量", "建议", "注意"]
            important_sentences = []
            other_sentences = []

            for sentence in unique_sentences:
                if any(keyword in sentence for keyword in important_keywords):
                    important_sentences.append(sentence)
                else:
                    other_sentences.append(sentence)

            # 优先保留重要句子，限制总长度
            if len(important_sentences) >= 5:
                unique_sentences = important_sentences[:5]
            else:
                unique_sentences = important_sentences + other_sentences[:5 - len(important_sentences)]

        cleaned_text = "。".join(unique_sentences)
        if cleaned_text and not cleaned_text.endswith("。"):
            cleaned_text += "。"

        # 3. 确保只有一个免责声明
        if "仅供参考" in cleaned_text:
            # 移除多余的免责声明
            cleaned_text = re.sub(r'以上信息仅供参考[^\n。]*[。\n]', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'请咨询专业医生[^\n。]*[。\n]', '', cleaned_text, flags=re.DOTALL)
            # 在末尾添加一个干净的免责声明
            if not cleaned_text.endswith("。") and not cleaned_text.endswith("."):
                cleaned_text += "。"
            cleaned_text += "\n\n⚠️ 以上信息仅供参考，不能替代专业医疗建议，请咨询医生。"
        else:
            cleaned_text += "\n\n⚠️ 以上信息仅供参考，不能替代专业医疗建议，请咨询医生。"

        return cleaned_text

    def answer_question(self,
                        query: str,
                        use_rag: bool = True,
                        include_references: bool = True) -> Dict:
        """
        回答用户问题 - 修复版
        """
        print(f"\n[处理] 问题: {query}")

        # 检索相关信息
        context = []
        if use_rag and self.vector_db is not None:
            context = self.retrieve_relevant_info(query, top_k=3)

        # 生成回答
        if self.llm_loaded and context:
            # 使用LLM + RAG
            prompt = self.create_prompt(query, context, self.conversation_history)
            answer = self.generate_answer_with_llm(prompt)
        elif context:
            # 只有RAG，没有LLM
            answer = self.generate_answer_from_context(query, context)
        else:
            # 回退模式
            answer = self.generate_fallback_answer(query)

        # 更新对话历史
        self.conversation_history.append((query, answer))
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        # 准备返回结果
        result = {
            "question": query,
            "answer": answer,
            "has_references": len(context) > 0,
            "retrieved_count": len(context)
        }

        # 添加参考信息
        if include_references and context:
            result["references"] = []
            for i, ref in enumerate(context, 1):
                result["references"].append({
                    "序号": i,
                    "相关性": f"{ref['score']:.3f}",
                    "科室": ref.get('department', '未知'),
                    "参考问题": ref['question'][:30] + ("..." if len(ref['question']) > 30 else ""),
                    "参考内容": ref['answer'][:50] + ("..." if len(ref['answer']) > 50 else "")
                })

        print(f"[完成] 回答长度: {len(answer)} 字符")
        return result

    def generate_answer_from_context(self, query: str, context: List[Dict]) -> str:
        """
        直接从上下文中生成回答
        """
        if not context:
            return self.generate_fallback_answer(query)

        # 使用最相关的上下文
        best_context = context[0]

        answer = f"根据医疗信息库：\n\n"
        answer += f"{best_context['answer']}\n\n"

        if len(context) > 1:
            answer += "其他相关信息：\n"
            for i, ctx in enumerate(context[1:3], 2):
                answer += f"{i}. {ctx['answer'][:50]}...\n"

        answer += "\n⚠️ 以上信息仅供参考，具体情况请咨询医生。"
        return answer

    def generate_fallback_answer(self, query: str) -> str:
        """
        生成回退回答
        """
        # 简单的关键词匹配
        fallback_responses = {
            "高血压": """💊 **高血压管理建议**：
1. **药物治疗**：需医生评估后选择合适的降压药
2. **生活方式**：低盐饮食、规律运动、控制体重
3. **监测**：定期测量血压，记录变化

📌 **注意**：具体用药方案需医生根据病情制定。""",

            "糖尿病": """💉 **糖尿病管理要点**：
1. **血糖控制**：定期监测血糖
2. **药物治疗**：口服降糖药或胰岛素
3. **饮食控制**：限制碳水化合物，多吃蔬菜
4. **运动**：每周至少150分钟中等强度运动

📌 **注意**：血糖控制目标因人而异。""",

            "感冒": """🤧 **感冒处理建议**：
1. **休息**：保证充足睡眠
2. **补水**：多喝温水
3. **对症用药**：解热镇痛药缓解症状
4. **就医**：如高热不退或症状加重，请及时就医

📌 **注意**：普通感冒多为病毒感染，抗生素无效。""",

            "胃痛": """🤢 **胃痛处理建议**：
1. **饮食**：清淡易消化，避免辛辣刺激
2. **药物**：可考虑胃黏膜保护剂或抗酸药
3. **就医**：如疼痛持续，请咨询消化内科医生

📌 **注意**：胃痛可能由多种原因引起，需明确诊断。"""
        }

        for keyword, response in fallback_responses.items():
            if keyword in query:
                return response

        return "这是一个医疗相关问题。建议咨询专业医生获取准确诊断。"

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []


def create_interface():
    """创建Gradio界面 - 新版布局：对话框在上，检索结果在下"""
    assistant = ImprovedMedicalAssistant()

    def respond(message, history, use_rag):
        if not message.strip():
            return history, "", []

        # 获取回答
        result = assistant.answer_question(message, use_rag=use_rag)

        # 更新聊天历史
        history.append((message, result["answer"]))

        # 准备参考信息 - 修复格式问题
        references = []
        if "references" in result and result["references"]:
            # 转换为二维列表格式
            for ref in result["references"]:
                references.append([
                    ref.get("序号", ""),
                    ref.get("相关性", ""),
                    ref.get("科室", ""),
                    ref.get("参考问题", ""),
                    ref.get("参考内容", "")
                ])

        print(f"[界面] 发送回答，参考信息数量: {len(references)}")
        return history, "", references

    def clear_chat():
        assistant.clear_history()
        return [], []

    with gr.Blocks(title="医疗问答系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏥 医疗问答系统
        """)

        # ========== 第一部分：对话框和输入 ==========
        with gr.Row():
            chatbot = gr.Chatbot(
                height=400,
                bubble_full_width=True,
                show_copy_button=True,
                label="医疗问答对话"
            )

        with gr.Row():
            user_input = gr.Textbox(
                placeholder="请输入医疗问题，如：高血压吃什么药？",
                lines=3,
                label="输入问题",
                scale=4
            )

            with gr.Column(scale=1):
                submit_btn = gr.Button("发送", variant="primary", size="lg")
                clear_btn = gr.Button("清空对话", variant="secondary")

        with gr.Row():
            rag_toggle = gr.Checkbox(
                label="启用智能检索(RAG)",
                value=True,
                info="启用后系统会从医疗知识库中检索相关信息"
            )

        gr.Markdown("---")  # 分隔线

        # ========== 第二部分：检索结果 ==========
        with gr.Row():
            gr.Markdown("### 🔍 检索结果")

        with gr.Row():
            references_display = gr.Dataframe(
                headers=["序号", "相关性", "科室", "参考问题", "参考内容"],
                datatype=["str", "str", "str", "str", "str"],
                height=300,
                wrap=True,
                interactive=False,
                label="检索到的相关信息"
            )

        # 状态信息
        with gr.Row():
            status_info = gr.Markdown(
                "**系统状态**: 就绪 | **知识库**: 610,742条数据 | **模型**: Qwen2.5-1.5B"
            )

        # 设置事件处理
        submit_btn.click(
            respond,
            inputs=[user_input, chatbot, rag_toggle],
            outputs=[chatbot, user_input, references_display]
        )

        user_input.submit(
            respond,
            inputs=[user_input, chatbot, rag_toggle],
            outputs=[chatbot, user_input, references_display]
        )

        clear_btn.click(
            clear_chat,
            outputs=[chatbot, references_display]
        )

        # 添加一些使用提示
        with gr.Accordion("💡 使用提示", open=False):
            gr.Markdown("""
            1. **输入问题**：在下方输入框输入您的医疗问题
            2. **发送方式**：点击"发送"按钮或按Enter键
            3. **检索功能**：启用RAG可以从知识库中检索相关信息
            4. **查看参考**：下方的表格显示检索到的相关信息
            5. **清空对话**：点击"清空对话"按钮可以开始新的对话
            6. **重要提醒**：所有回答仅供参考，请咨询专业医生
            """)

    return demo


def main():
    print("启动医疗问答系统...")
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()