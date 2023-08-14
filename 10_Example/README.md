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


## 总结

本节课程中，我们利用所学的知识，完成了第一个完整的LLM应用。希望通过本系列的学习，大家能对 `LangChain` 框架的使用，有了基本的认识，并且掌握了框架核心组建的使用方法。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/get_started/introduction.html) 