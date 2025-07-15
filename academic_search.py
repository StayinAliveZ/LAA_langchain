import os
import tempfile
import requests
import json
from langchain.tools import tool
from unstructured.partition.pdf import partition_pdf
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Step 1: Helper Function to Extract Title and Raw Text ---

def extract_title_and_initial_text(pdf_file):
    """
    从PDF中提取标题和用于摘要生成的初始原始文本。
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.getvalue())
            tmp_path = tmp.name

        try:
            print("Attempting 'hi_res' PDF partitioning...")
            elements = partition_pdf(tmp_path, strategy="hi_res", infer_table_structure=True)
        except Exception as e:
            print(f"Warning: 'hi_res' strategy failed: {e}. Falling back to 'fast' strategy.")
            elements = partition_pdf(tmp_path, strategy="fast")

        # --- Title Extraction ---
        title = "Unknown Title"
        title_element_index = -1
        for i, element in enumerate(elements):
            if element.category == "Title":
                title = element.text
                title_element_index = i
                break
        if title == "Unknown Title" and elements:
            for i, element in enumerate(elements):
                potential_title = element.text.strip()
                if potential_title and len(potential_title) < 200 and '\n' not in potential_title:
                    title = potential_title
                    title_element_index = i
                    break
        
        # --- Raw Text Extraction for Summarization ---
        initial_text = ""
        start_index = title_element_index + 1 if title_element_index != -1 else 0
        for el in elements[start_index:]:
            text = el.text.strip()
            if text:
                initial_text += text + "\n"
            if len(initial_text) > 3500: # Limit text to avoid excessive LLM costs/time
                break
        
        if not initial_text:
            return {"title": title, "initial_text": "Could not extract any text from the document."}

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    return {"title": title.strip(), "initial_text": initial_text.strip()}


# --- LangChain Tools for the Agent ---

# --- Step 2: Tool to Summarize Text into an Abstract ---

class SummarizeInput(BaseModel):
    paper_text: str = Field(description="需要被总结成摘要的论文原始文本。")

@tool("summarize-text-to-abstract", args_schema=SummarizeInput)
def summarize_text_to_abstract(paper_text: str) -> str:
    """
    将论文的原始文本总结成一段简洁的、学术风格的摘要。
    """
    llm = init_chat_model("deepseek-chat", model_provider="deepseek")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位专业的学术编辑。你的任务是阅读所提供的论文初始文本，并将其总结为一段约150词的、简洁专业的学术摘要。请捕捉文本中提出的核心问题、使用的方法、主要结果和结论。"),
        ("human", "请将以下文本总结成一段摘要：\n\n{text}"),
    ])
    chain = prompt | llm
    response = chain.invoke({"text": paper_text})
    print(f"Generated Abstract: {response.content}")
    return response.content

# --- Step 3: Tool to Generate Keywords ---

class KeywordGeneratorInput(BaseModel):
    title: str = Field(description="论文的标题")
    abstract: str = Field(description="论文的摘要内容（通常由summarize-text-to-abstract工具生成）")

@tool("generate-search-keywords", args_schema=KeywordGeneratorInput)
def generate_search_keywords(title: str, abstract: str) -> list[str]:
    """
    根据论文的标题和摘要，提炼出3-5个最核心的英文学术检索关键词。
    """
    llm = init_chat_model("deepseek-chat", model_provider="deepseek")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位精通信息检索的图书情报专家。你的任务是根据给定的论文标题和摘要，提炼出3-5个最核心、最简洁的学术检索关键词。重要提示：这些关键词将用于国际学术搜索引擎，因此无论输入摘要是何种语言，输出的关键词都必须是英文的。请务必以一个只包含JSON对象的格式返回，格式为：\n{{\"keywords\": [\"english_keyword_1\", \"english_keyword_2\", ...]}}。不要包含任何JSON格式之外的额外文本或解释。"),
        ("human", "标题: {title}\n\n摘要: {abstract}"),
    ])
    
    chain = prompt | llm
    try:
        response_content = chain.invoke({"title": title, "abstract": abstract}).content
    except Exception as e:
        print(f"Error invoking LLM for keyword generation: {e}")
        return [] # On failure, return an empty list to prevent the agent from crashing

    try:
        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0].strip()
        elif response_content.startswith("```"):
             response_content = response_content[3:-3].strip()

        data = json.loads(response_content)
        keywords = data.get("keywords", [])

        if isinstance(keywords, list) and all(isinstance(kw, str) for kw in keywords):
            print(f"Generated Keywords via JSON: {keywords}")
            return keywords[:5]
        else:
            return []
            
    except json.JSONDecodeError:
        print(f"Warning: Failed to decode JSON. Falling back to string parsing. Response: {response_content}")
        keywords = [kw.strip() for kw in response_content.split(',') if kw.strip() and len(kw.split()) < 5]
        return keywords[:5]

# --- Step 4: Tool to Search Papers ---

class SemanticScholarInput(BaseModel):
    keywords: list[str] = Field(description="用于在Semantic Scholar上检索的关键词列表")

@tool("search-semantic-scholar", args_schema=SemanticScholarInput)
def search_semantic_scholar(keywords: list[str]) -> list[dict]:
    """
    使用关键词列表在Semantic Scholar上搜索相关学术论文。
    """
    if not keywords:
        return []

    query = " ".join(keywords)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,year,authors,abstract,url"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            error_details = response.json().get('error', 'No specific error message provided.')
            print(f"Error calling Semantic Scholar API: {response.status_code} - {error_details}")
            return [{"error": f"API request failed with status {response.status_code}: {error_details}"}]

        results = response.json()
        papers = results.get("data", [])
        
        formatted_papers = []
        for paper in papers:
            authors_list = paper.get('authors', [])
            authors = ", ".join([author['name'] for author in authors_list if author and 'name' in author])
            formatted_papers.append({
                "title": paper.get("title", "N/A"),
                "authors": authors,
                "year": paper.get("year", "N/A"),
                "abstract": paper.get("abstract", "N/A"),
                "url": paper.get("url", "#")
            })
        print(f"Found {len(formatted_papers)} papers from Semantic Scholar.")
        return formatted_papers

    except requests.RequestException as e:
        print(f"Error calling Semantic Scholar API: {e}")
        return [{"error": f"API request failed: {e}"}]
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        return [{"error": f"An unexpected error occurred: {e}"}] 