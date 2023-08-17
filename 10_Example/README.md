---
title: 10. 一个完整的例子
tags:
  - openai
  - llm
  - langchain
---

# WTF Langchain极简入门: 10. 一个完整的例子

最近在学习Langchain框架，顺手写一个“WTF Langchain极简入门”，供小白们使用（编程大佬可以另找教程）。本教程默认以下前提：
- 使用Python版本的[Langchain](https://github.com/hwchase17/langchain)
- LLM使用OpenAI的模型
- Langchain目前还处于快速发展阶段，版本迭代频繁，为避免示例代码失效，本教程统一使用版本 **0.0.235**

根据Langchain的[代码约定](https://github.com/hwchase17/langchain/blob/v0.0.235/pyproject.toml#L14C1-L14C24)，Python版本 ">=3.8.1,<4.0"。

推特：[@verysmallwoods](https://twitter.com/verysmallwoods)

所有代码和教程开源在github: [github.com/sugarforever/wtf-langchain](https://github.com/sugarforever/wtf-langchain)

-----

## 简介

这是该 `LangChain` 极简入门系列的最后一讲。我们将利用过去9讲学习的知识，来完成一个具备完整功能集的LLM应用。该应用基于 `LangChain` 框架，以某 `PDF` 文件的内容为知识库，提供给用户基于该文件内容的问答能力。

我们利用 `LangChain` 的QA chain，结合 `Chroma` 来实现PDF文档的语义化搜索。示例代码所引用的是[AWS Serverless
Developer Guide](https://docs.aws.amazon.com/pdfs/serverless/latest/devguide/serverless-core.pdf)，该PDF文档共84页。

本讲的完整代码请参考[10_Example.jpynb](./10_Example.ipynb)

1. 安装必要的 `Python` 包

    ```shell
    !pip install -q langchain==0.0.235 openai chromadb pymupdf tiktoken
    ```

2. 设置OpenAI环境

    ```python
    import os
    os.environ['OPENAI_API_KEY'] = '您的有效openai api key'
    ```

3. 下载PDF文件AWS Serverless Developer Guide

    ```python
    !wget https://docs.aws.amazon.com/pdfs/serverless/latest/devguide/serverless-core.pdf

    PDF_NAME = 'serverless-core.pdf'
    ```

4. 加载PDF文件

    ```python
    from langchain.document_loaders import PyMuPDFLoader
    docs = PyMuPDFLoader(PDF_NAME).load()

    print (f'There are {len(docs)} document(s) in {PDF_NAME}.')
    print (f'There are {len(docs[0].page_content)} characters in the first page of your document.')
    ```

5. 拆分文档并存储文本嵌入的向量数据

    ```python
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import Chroma

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = Chroma.from_documents(split_docs, embeddings, collection_name="serverless_guide")
    ```

6. 基于OpenAI创建QA链

    ```python
    from langchain.llms import OpenAI
    from langchain.chains.question_answering import load_qa_chain

    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    ```

7. 基于提问，进行相似性查询
    
    ```python
    query = "What is the use case of AWS Serverless?"
    similar_docs = vectorstore.similarity_search(query, 3, include_metadata=True)
    ```

8. 基于相关文档，利用QA链完成回答

    ```python
    chain.run(input_documents=similar_docs, question=query)
    ```

## 总结

本节课程中，我们利用所学的知识，完成了第一个完整的LLM应用。希望通过本系列的学习，大家能对 `LangChain` 框架的使用，有了基本的认识，并且掌握了框架核心组建的使用方法。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/get_started/introduction.html) 