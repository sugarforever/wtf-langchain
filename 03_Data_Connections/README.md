---
title: 03. 数据连接
tags:
  - openai
  - llm
  - langchain
---

# WTF Langchain极简入门: 03. 数据连接

最近在学习Langchain框架，顺手写一个“WTF Langchain极简入门”，供小白们使用（编程大佬可以另找教程）。本教程默认以下前提：
- 使用Python版本的[Langchain](https://github.com/hwchase17/langchain)
- LLM使用OpenAI的模型
- Langchain目前还处于快速发展阶段，版本迭代频繁，为避免示例代码失效，本教程统一使用版本 **0.0.235**

根据Langchain的[代码约定](https://github.com/hwchase17/langchain/blob/v0.0.235/pyproject.toml#L14C1-L14C24)，Python版本 ">=3.8.1,<4.0"。

推特：[@verysmallwoods](https://twitter.com/verysmallwoods)

所有代码和教程开源在github: [github.com/sugarforever/wtf-langchain](https://github.com/sugarforever/wtf-langchain)

-----

## 什么是数据连接？

LLM应用往往需要用户特定的数据，而这些数据并不属于模型的训练集。`LangChain` 的数据连接概念，通过提供以下组件，实现用户数据的加载、转换、存储和查询：

- 文档加载器：从不同的数据源加载文档
- 文档转换器：拆分文档，将文档转换为问答格式，去除冗余文档，等等
- 文本嵌入模型：将非结构化文本转换为浮点数数组表现形式，也称为向量
- 向量存储：存储和搜索嵌入数据（向量）
- 检索器：提供数据查询的通用接口

我们通过下一段落的实践，来介绍这些组件的使用。
## 数据连接实践

在LLM应用连接用户数据时，通常我们会以如下步骤完成：
1. 加载文档
2. 拆分文档
3. 向量化文档分块
4. 向量数据存储

这样，我们就可以通过向量数据的检索器，来查询用户数据。接下来我们看看每一步的代码实现示例。最后，我们将通过一个完整的示例来演示如何使用数据连接。

### 加载文档

`Langchain` 提供了多种文档加载器，用于从不同的数据源加载不同类型的文档。比如，我们可以从本地文件系统加载文档，也可以通过网络加载远程数据。想了解 `Langchain` 所支持的所有文档加载器，请参考[Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/)。

在本节课程中，我们将使用最基本的 `TextLoader` 来加载本地文件系统中的文档。代码如下：

```python
from langchain.document_loaders import TextLoader

loader = TextLoader("./01_Hello_Langchain/README.md")
docs = loader.load()
```

在上述代码中，我们使用 `TextLoader` 加载了本地文件系统中的 `./01_Hello_Langchain/README.md` 文件。`TextLoader` 的 `load` 方法返回一个 `Document` 对象数组（`Document` 是 `Langchain` 提供的文档类，包含原始内容和元数据）。我们可以通过 `Document` 对象的 `content` 属性来访问文档的原始内容。

### 拆分文档

拆分文档是文档转换中最常见的操作。拆分文档的目的是将文档拆分为更小的文档块，以便更好地利用模型。当我们要基于长篇文本构建QA应用时，必须将文本分割成块，这样才能在数据查询中，基于相似性找到与问题最接近的文本块。这也是常见的AI问答机器人的工作原理。

`Langchain` 提供了多种文档拆分器，用于将文档拆分为更小的文档块。我们逐个看看这些拆分器的使用方法。

#### 按字符拆分

`CharacterTextSplitter` 是最简单的文档拆分器，它将文档拆分为固定长度的文本块。代码如下：

```python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(        
    separator = "\n\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)

split_docs = text_splitter.split_documents(docs)
```

#### 拆分代码

`RecursiveCharacterTextSplitter` 的 `from_language` 函数，可以根据编程语言的特性，将代码拆分为合适的文本块。代码如下：

```pyhon
PYTHON_CODE = """
def hello_langchain():
    print("Hello, Langchain!")

# Call the function
hello_langchain()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs
```

#### Markdown文档拆分

`MarkdownHeaderTextSplitter` 可以将Markdown文档按照段落结构，基于Markdown语法进行文档分块。代码如下：

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_document = "# Chapter 1\n\n    ## Section 1\n\nHi this is the 1st section\n\nWelcome\n\n ### Module 1 \n\n Hi this is the first module \n\n ## Section 2\n\n Hi this is the 2nd section"

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = splitter.split_text(markdown_document)
```

#### 按字符递归拆分

这也是对于普通文本的推荐拆分方式。它通过一组字符作为参数，按顺序尝试使用这些字符进行拆分，直到块的大小足够小为止。默认的字符列表是["\n\n", "\n", " ", ""]。它尽可能地保持段落，句子，单词的完整，因此尽可能地保证语义完整。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 100,
    chunk_overlap  = 20,
    length_function = len,
)
texts = text_splitter.split_documents(docs)
```
#### 按token拆分

语言模型，例如OpenAI，有token限制。在API调用中，不应超过token限制。看来，在将文本分块时，根据token数来拆分也是不错的主意。有许多令牌化工具，因此当计算文本的token数时，应该使用与语言模型相同的令牌化工具。

注，OpenAI所使用的是[tiktoken](https://github.com/openai/tiktoken)。

```python
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)
```
### 向量化文档分块

`Langchain` 的 `Embeddings` 类实现与文本嵌入模型进行交互的标准接口。当前市场上有许多嵌入模型提供者（如OpenAI、Cohere、Hugging Face等）。

嵌入模型创建文本片段的向量表示。这意味着我们可以在向量空间中处理文本，并进行语义搜索等操作，从而查找在向量空间中最相似的文本片段。

注，文本之间的相似度由其向量表示的欧几里得距离来衡量。欧几里得距离是最常见的距离度量方式，也称为L2范数。对于两个n维向量a和b，欧几里得距离可以通过以下公式计算：

```math
d(a, b) = √((a₁ - b₁)² + (a₂ - b₂)² + ... + (aₙ - bₙ)²)
```

当使用OpenAI的文本嵌入模型时，我们使用如下代码来创建文本片段的向量表示：

```python
from langchain.embeddings import OpenAIEmbeddings
embeddings_model = OpenAIEmbeddings(openai_api_key="你的有效openai api key")
embeddings = embeddings_model.embed_documents(
    [
        "你好!",
        "Langchain!",
        "你真棒！"
    ]
)
embeddings
```

你应该能看到如下输出：

```shell
[[0.001767348474591444,
  -0.016549955833298362,
  0.009669921232251705,
  -0.024465152668289573,
  -0.04928377577655549,
  ...],
  [...
  -0.026084684286027195,
  -0.0023797790465254626,
  0.006104789779720747,
  ...]]
```

注，当我们在环境变量中设置了 `OPENAI_API_KEY`，在上述代码 `OpenAIEmbeddings(openai_api_key="你的有效openai api key")` 中，参数 `openai_api_key` 可以省略。

```shell
export OPENAI_API_KEY="..."
```

### 向量数据存储

向量数据存储，或成为向量数据库，负责存储文本嵌入的向量表现，并提供向量检索的能力。`Langchain` 提供了多种开源或商业向量数据存储，包括：Chroma, FAISS, Pinecone等。

本讲以Chroma为例。

#### 存储

`Langchain` 提供了 `Chroma` 包装类，封装了chromadb的操作。

在进行以下代码执行前，需要安装 `Chroma` 的包：

```shell
pip install -q chromadb
```

```python
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(docs)
db = Chroma.from_documents(documents, OpenAIEmbeddings())
```

#### 检索

向量数据库提供的重要接口就是相似性查询。如上述内容提到，文本相似性的衡量，由文本的向量表示的欧几里得距离来衡量。向量数据库提供了该接口，用于查询与给定文本最相似的文本。

```python
query = "什么是WTF Langchain？"
docs = db.similarity_search(query)
print(docs[0].page_content)
```

## 总结
本节课程中，我们简要介绍了 `Langchain` 的数据连接概念，并完成了以下任务：
1. 了解常见的文档加载器，
2. 了解常见的文档拆分方法
3. 掌握如何利用OpenAI实现文本的向量化和向量数据的存储与查询。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/get_started/introduction.html) 
2. [本节课程的示例代码](./03_Data_Connections.ipynb)