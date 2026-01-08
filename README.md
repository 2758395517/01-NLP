# 医疗问答系统（RAG-based Medical Q&A System）

一个基于检索增强生成（RAG）架构的中文医疗智能问答系统，集成了大规模医疗知识库与深度学习模型，为用户提供医疗信息查询服务。

## 项目特点
- **大规模知识库**：基于792,099条中文医疗QA数据构建
- **智能检索**：采用m3e-base模型进行语义向量检索
- **专业生成**：集成Qwen2.5-1.5B-Instruct模型生成专业回答
- **可解释性**：提供检索来源参考，增强回答可信度
- **用户友好**：基于Gradio的交互式Web界面

## 数据来源
数据集链接：https://github.com/Toyhom/Chinese-medical-dialogue-data

## 项目结构
NLP/  
├── data_数据 # 数据集文件  
├── data.py # 数据处理与清洗模块  
├── database.py # 向量数据库构建与管理  
├── LLM.py # LLM集成与问答系统核心  
├── requirements.txt # 依赖包列表  
└── README.md # 项目说明文档  

## 环境要求
- Python 3.8
- Anaconda+pycharm

## 安装步骤
1. **克隆项目**
```bash
git clone https://github.com/2758395517/01-NLP.git
cd 01-NLP
```
2. **创建虚拟环境**
```bash
conda create -n medical-qa python=3.8
```
3. **安装其他依赖**
```bash
pip install -r requirements.txt
```
## 运行步骤
**第一步：准备数据**
将医疗数据CSV文件放置在Data_数据目录中，然后运行：
```bash
python data.py
```
此步骤将：
自动检测并处理不同编码格式、清洗和标准化数据、生成medical_chunks.json文件

**第二步：构建向量数据库**
```bash
python database.py
```
此步骤将：使用m3e-base模型生成文本向量、构建FAISS向量索引、保存为medical_vector_db系列文件

**第三步：启动Web服务**
```bash
python LLM.py
```
系统将在本地启动，访问地址：http://127.0.0.1:7860
