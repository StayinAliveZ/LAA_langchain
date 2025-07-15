import streamlit as st
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import tempfile
import ast # Import the Abstract Syntax Tree module
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.chat_models import init_chat_model
import os
import time
from dotenv import load_dotenv

# Import tools and functions from our new module
from academic_search import (
    extract_title_and_initial_text,
    summarize_text_to_abstract,
    generate_search_keywords,
    search_semantic_scholar,
)

load_dotenv(override=True)

# 设置Hugging Face镜像，解决模型下载问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

DeepSeek_API_KEY = os.getenv("DEEPSEEK_API_KEY")
dashscope_api_key = os.getenv("dashscope_api_key")

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def inject_custom_css():
    st.markdown("""
        <style>
            /* General App Styling */
            .stApp {
                background-color: #F0F2F6; /* A light, neutral background */
            }

            /* Main Title */
            h1 {
                color: #262730;
                font-family: "Source Sans Pro", sans-serif;
                font-weight: 700;
                text-align: center;
                padding-bottom: 10px;
                margin-bottom: 20px;
                border-bottom: 2px solid #D1D5DB;
            }

            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background-color: #FFFFFF;
                border-right: 1px solid #E5E7EB;
            }
            [data-testid="stSidebar"] h1 {
                font-size: 24px;
                text-align: left;
                border-bottom: none;
                padding-bottom: 0;
                margin-bottom: 15px;
            }

            /* Custom button styling */
            .stButton>button {
                border: 1px solid #BCCCDC; /* Softer border color */
                border-radius: 8px;
                color: #263238; /* Darker text for readability */
                background-color: #FFFFFF;
                padding: 10px 24px;
                font-weight: 600;
                transition: all 0.2s ease-in-out;
            }
            .stButton>button:hover {
                border-color: #374151;
                color: #FFFFFF;
                background-color: #4A5568; /* A cool gray on hover */
            }
            .stButton>button:disabled {
                border: 1px solid #D1D5DB;
                color: #9CA3AF;
                background-color: #F3F4F6;
            }
            
            /* Main action button in sidebar */
            [data-testid="stSidebar"] .stButton[data-testid="stButton"] button {
                 background-color: #374151;
                 color: white;
                 width: 100%;
            }
             [data-testid="stSidebar"] .stButton[data-testid="stButton"] button:hover {
                 background-color: #262730;
             }

            /* Text Input Box */
            .stTextInput>div>div>input {
                border: 1px solid #D1D5DB;
                background-color: #FFFFFF;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
            }
            .stTextInput>div>div>input:focus {
                border: 2px solid #374151;
                box-shadow: none;
            }
            
            /* Expander (Usage Guide) */
            .stExpander {
                border: 1px solid #E5E7EB;
                box-shadow: none;
                border-radius: 8px;
                background-color: #FFFFFF;
            }
            .stExpander header {
                font-size: 16px;
                font-weight: 600;
                color: #374151;
            }

            /* Status boxes (Success, Warning, Info) */
            [data-testid="stAlert"] {
                border-radius: 8px;
                padding-left: 20px;
                background-color: #FFFFFF;
                border-left: 5px solid;
            }
            [data-testid="stAlert"][data-baseweb="notification-positive"] {
                border-color: #34D399;
                color: #065F46;
            }
            [data-testid="stAlert"][data-baseweb="notification-warning"] {
                border-color: #FBBF24;
                color: #92400E;
            }
            [data-testid="stAlert"][data-baseweb="notification-info"] {
                border-color: #60A5FA;
                color: #1E40AF;
            }
        </style>
    """, unsafe_allow_html=True)


embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key=dashscope_api_key
)

def process_pdfs(pdf_docs):
    """
    使用unstructured库处理PDF文件，进行结构化切分并存入向量数据库。
    此版本使用 'chunk_by_title' 策略来创建更符合逻辑、更大的块。
    """
    all_docs = []
    for pdf in pdf_docs:
        # Unstructured需要一个文件路径。我们将上传的文件保存到一个临时文件中。
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.getvalue())
            tmp_path = tmp.name
        
        # 1. 使用unstructured的partition_pdf智能提取元素
        # strategy="hi_res" 使用高精度模型，能更好地识别标题、表格等，但需要更多计算资源，且需要下载模型
        elements = partition_pdf(tmp_path, strategy="fast", infer_table_structure=True)
        
        # 2. 使用chunk_by_title策略将元素分块
        # 这个策略会根据文档中的标题（<h1>, <h2>等）来组织内容，非常适合学术论文
        # max_characters 控制每个块的最大字符数，以适应模型的上下文窗口
        chunks = chunk_by_title(
            elements,
            max_characters=2000,      # 增加每个块的最大字符数
            new_after_n_chars=1500,   # 在此字符数后倾向于创建新块
            combine_text_under_n_chars=500  # 合并小于此字符数的文本块
        )

        # 3. 将分块结果转换为LangChain的Document对象
        for chunk in chunks:
            # 将unstructured的Element对象转换为dict
            metadata = chunk.metadata.to_dict()
            # 移除一些不需要或可能导致问题的元数据
            metadata.pop('coordinates', None)
            metadata.pop('parent_id', None)
            # 添加文件来源信息
            metadata['source'] = pdf.name
            
            all_docs.append(Document(page_content=chunk.text, metadata=metadata))
        
        # 清理临时文件
        os.remove(tmp_path)
    
    # 在处理完所有PDF后，创建向量数据库
    if not all_docs:
        raise ValueError("无法从PDF文件中提取任何内容，请检查文件是否有效。")
        
    vector_store(all_docs)
    return len(all_docs)


def vector_store(documents):
    # loader直接提供了Document对象，因此我们使用from_documents
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_db")

def get_conversational_chain(tools, ques):
    llm = init_chat_model("deepseek-chat", model_provider="deepseek")
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是一位专业的学术研究助理。你的任务是根据提供的上下文（学术论文），精准、清晰地回答用户的问题。请严格遵守以下指南：
1.  **综合信息**：从文档的各个部分综合信息，提供全面而深入的回答，而不仅仅是列出找到的片段。
2.  **精准具体**：当被问及研究方法、创新点、结论或数据时，提供具体细节和文本中的证据。
3.  **承认局限**：如果论文中没有相关信息或表述模糊，请明确指出，不要杜撰答案。
4.  **结构化回答**：对于复杂问题，使用要点、列表或段落来组织回答，以提高可读性。
5.  **忠于原文**：所有回答都必须严格基于所提供的PDF文档内容。如果答案不在上下文中，请说“答案不在提供的文献中”。""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    tool = [tools]
    agent = create_tool_calling_agent(llm, tool, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    
    response = agent_executor.invoke({"input": ques})
    print(response)
    st.write("🤖 回答: ", response['output'])

def check_database_exists():
    """检查FAISS数据库是否存在"""
    return os.path.exists("faiss_db") and os.path.exists("faiss_db/index.faiss")

def user_input(user_question):
    # 检查数据库是否存在
    if not check_database_exists():
        st.error("❌ 请先上传PDF文件并点击'Submit & Process'按钮来处理文档！")
        st.info("💡 步骤：1️⃣ 上传PDF → 2️⃣ 点击处理 → 3️⃣ 开始提问")
        return
    
    try:
        # 加载FAISS数据库
        new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
        # 可以修改top_k的值，来控制返回的文档数量，默认是4。个人推荐k=6的效果比较好
        retriever = new_db.as_retriever(search_kwargs={'k': 6})
        retrieval_chain = create_retriever_tool(retriever, "pdf_extractor", "This tool is to give answer to queries from the pdf")
        get_conversational_chain(retrieval_chain, user_question)
        
    except Exception as e:
        st.error(f"❌ 加载数据库时出错: {str(e)}")
        st.info("请重新处理PDF文件")


def run_related_papers_agent(pdf_file):
    """
    运行一个Agent来查找与给定PDF相关的文献。
    """
    with st.spinner("正在提取论文核心信息..."):
        try:
            details = extract_title_and_initial_text(pdf_file)
            title = details.get("title")
            initial_text = details.get("initial_text")

            if title == "Unknown Title":
                st.error("无法从此PDF中识别出标题，请尝试其他文件。")
                return
            st.info(f"以论文 **《{title}》** 为基础进行分析...")

        except Exception as e:
            st.error(f"解析PDF时出错: {e}")
            return

    with st.spinner("AI正在生成检索策略并查找相关文献..."):
        try:
            # 1. 定义可用的工具
            tools = [summarize_text_to_abstract, generate_search_keywords, search_semantic_scholar]

            # 2. 创建LLM
            llm = init_chat_model("deepseek-chat", model_provider="deepseek")

            # 3. 创建Agent的提示词 (包含新的三步工作流)
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个强大的科研助理。你的任务是根据提供的论文标题和初始文本，找到相关的学术文献。你必须严格遵循以下步骤：
1.  **第一步：总结摘要**。使用`summarize-text-to-abstract`工具，为提供的`initial_text`生成一段简洁的摘要。
2.  **第二步：提炼关键词**。使用`generate-search-keywords`工具，并为其提供原始的`title`和你刚刚在第一步生成的`abstract`。
3.  **第三步：搜索文献**。使用`search-semantic-scholar`工具，并为其提供你在第二步生成的`keywords`。
4.  **最后**，你必须将`search-semantic-scholar`工具返回的原始Python列表（list of dictionaries）作为你的最终答案。不要对它进行任何格式化、转换或添加任何解释性文字。必须直接返回原始的列表结构。"""),
                ("human", "请为以下文档查找相关文献：\n\n标题: {title}\n\n初始文本: {initial_text}"),
                ("placeholder", "{agent_scratchpad}"),
            ])

            # 4. 创建Agent
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            # 5. 运行Agent
            response = agent_executor.invoke({
                "title": title,
                "initial_text": initial_text
            })
            
            # 6. 解析并显示结果 (关键修正)
            output = response.get('output')
            
            if isinstance(output, str):
                try:
                    # AI可能返回一个列表的字符串表示，我们安全地将其解析为真实的列表
                    st.session_state.related_papers = ast.literal_eval(output)
                except (ValueError, SyntaxError):
                    # 如果解析失败，说明它不是一个合法的列表字符串
                    print(f"Agent返回了无法解析的字符串: {output}")
                    st.session_state.related_papers = [] # 设为空列表以显示失败信息
            elif isinstance(output, list):
                # AI正确地返回了一个列表
                st.session_state.related_papers = output
            else:
                # 任何其他意外类型
                print(f"Agent返回了意外的类型: {type(output)}")
                st.session_state.related_papers = []
        
        except Exception as e:
            st.error(f"查找相关文献时出错: {e}")
            st.session_state.related_papers = [] # 在发生异常时也确保清空


def main():
    st.set_page_config("📚 学术文献分析助手", layout="wide")
    inject_custom_css()
    
    st.title("🔬 学术洞察引擎")
    
    # 显示数据库状态
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if check_database_exists():
            st.success("✅ 知识库已就绪，可以开始提问。")
        else:
            st.warning("⚠️ 请先上传并处理PDF文件以构建知识库。")
    
    with col2:
        if st.button("🗑️ 清除知识库"):
            try:
                import shutil
                if os.path.exists("faiss_db"):
                    shutil.rmtree("faiss_db")
                st.success("知识库已清除！")
                time.sleep(3) # 暂停3秒
                st.rerun()
            except Exception as e:
                st.error(f"清除失败: {e}")

    # 用户问题输入
    user_question = st.text_input("💬 针对文献，提出您的问题", 
                                placeholder="例如：总结一下这篇论文的核心贡献。这篇论文的创新点是什么？",
                                disabled=not check_database_exists())

    if user_question:
        if check_database_exists():
            with st.spinner("🤔 AI正在分析文献，请稍候..."):
                user_input(user_question)
        else:
            st.error("❌ 请先上传并处理PDF文件！")

    # Display related papers if they exist in the session state
    if "related_papers" in st.session_state and st.session_state.related_papers:
        st.markdown("---")
        st.subheader("🔍 相关文献推荐")
        
        with st.expander("点击查看推荐的5篇相关文献", expanded=True):
            papers = st.session_state.related_papers
            if isinstance(papers, list) and papers:
                for i, paper in enumerate(papers):
                    if "error" in paper:
                        st.error(f"在检索过程中发生错误: {paper['error']}")
                        continue

                    st.markdown(f"**{i+1}. {paper.get('title', 'N/A')}**")
                    st.markdown(f"_作者: {paper.get('authors', 'N/A')} ({paper.get('year', 'N/A')})_")
                    st.markdown(f"**摘要**: {paper.get('abstract', '无可用摘要')}")
                    st.markdown(f"[🔗 阅读原文]({paper.get('url', '#')})")
                    if i < len(papers) - 1:
                        st.markdown("---")
            else:
                st.write("未能找到相关的文献，或返回格式不正确。")

    # 侧边栏
    with st.sidebar:
        st.title("📚 文档库管理")
        
        # 显示当前状态
        if check_database_exists():
            st.success("✅ 状态：知识库已就绪")
        else:
            st.info("ℹ️ 状态：等待上传文献")
        
        st.markdown("---")
        
        # 文件上传
        pdf_docs = st.file_uploader(
            "📎 上传PDF格式的学术文献", 
            accept_multiple_files=True,
            type=['pdf'],
            help="支持一次性上传多篇相关文献进行综合分析"
        )
        
        if pdf_docs:
            st.info(f"📄 已选择 {len(pdf_docs)} 篇文献")
            for i, pdf in enumerate(pdf_docs, 1):
                st.write(f"   {i}. {pdf.name}")
        
        st.markdown("---")

        # 处理按钮
        process_button = st.button(
            "🚀 构建知识库", 
            disabled=not pdf_docs,
            use_container_width=True
        )

        if process_button:
            if "related_papers" in st.session_state:
                del st.session_state.related_papers  # Clear previous recommendations
            if pdf_docs:
                with st.spinner("📊 正在处理文献，请稍候..."):
                    try:
                        # 新的处理流程
                        chunk_count = process_pdfs(pdf_docs)
                        st.info(f"✅ 文献已成功处理，切分为 {chunk_count} 个知识片段。")
                        st.success("🎉 知识库构建完成！现在可以开始提问了。")
                        st.balloons()
                        time.sleep(1.5) # 暂停1.5秒
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ 处理PDF时出错: {str(e)}")
            else:
                st.warning("⚠️ 请先选择要上传的PDF文件")
        
        # 新增：相关文献推荐功能
        st.markdown("---")
        st.subheader("🔗 相关文献推荐")

        if pdf_docs:
            pdf_options = {pdf.name: pdf for pdf in pdf_docs}
            selected_pdf_name = st.selectbox(
                "请选择一篇论文作为分析基础：",
                options=list(pdf_options.keys())
            )
            
            find_related_button = st.button(
                "查找相关文献",
                use_container_width=True
            )

            if find_related_button and selected_pdf_name:
                selected_pdf_file = pdf_options[selected_pdf_name]
                if "related_papers" in st.session_state:
                     del st.session_state.related_papers # Clear previous results before new search
                run_related_papers_agent(selected_pdf_file)
                st.rerun()
        else:
            st.info("请先上传PDF文件以启用此功能。")


        # 使用说明
        with st.expander("💡 使用指南"):
            st.markdown("""
            **分析流程：**
            1. 📎 **上传文献**：在上方上传一篇或多篇PDF格式的学术文献。
            2. 🚀 **构建知识库**：点击“构建知识库”按钮，AI将对文献进行深度解析和索引。
            3. 💬 **开始提问**：在主界面的输入框中，针对文献内容提出您的问题。
            4. 🤖 **获取洞察**：AI将基于文献内容，为您提供精准、综合的回答。
            
            **高级技巧：**
            - **多文件分析**：可同时上传多篇相关主题的论文，进行跨文献的综合问答。
            - **清除知识库**：当您想分析新的主题时，可以清除旧的知识库，重新开始。
            """)
            
        st.markdown("---")
        st.info("助手 by Zhang.E.B")

if __name__ == "__main__":
    main()
