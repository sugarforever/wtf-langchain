---
title: 02. 模型
tags:
  - openai
  - llm
  - langchain
---

# WTF Langchain极简入门: 02. 模型

最近在学习Langchain框架，顺手写一个“WTF Langchain极简入门”，供小白们使用（编程大佬可以另找教程）。本教程默认以下前提：
- 使用Python版本的[Langchain](https://github.com/hwchase17/langchain)
- LLM使用OpenAI的模型
- Langchain目前还处于快速发展阶段，版本迭代频繁，为避免示例代码失效，本教程统一使用版本 **0.0.235**

根据Langchain的[代码约定](https://github.com/hwchase17/langchain/blob/v0.0.235/pyproject.toml#L14C1-L14C24)，Python版本 ">=3.8.1,<4.0"。

推特：[@verysmallwoods](https://twitter.com/verysmallwoods)

所有代码和教程开源在github: [github.com/sugarforever/wtf-langchain](https://github.com/sugarforever/wtf-langchain)

-----

## 模型简介

Langchain所封装的模型分为两类：
- 大语言模型 (LLM)
- 聊天模型 (Chat Models)

在后续的内容中，为简化描述，我们将使用 `LLM` 来指代大语言模型。

Langchain的支持众多模型供应商，包括OpenAI、ChatGLM、HuggingFace等。本教程中，我们将以OpenAI为例，后续内容中提到的模型默认为OpenAI提供的模型。

Langchain的封装，比如，对OpenAI模型的封装，实际上是指的是对OpenAI API的封装。

### LLM

`LLM` 是一种基于统计的机器学习模型，用于对文本数据进行建模和生成。LLM学习和捕捉文本数据中的语言模式、语法规则和语义关系，以生成连贯并合乎语言规则的文本。

在Langchain的环境中，LLM特指文本补全模型（text completion model）。

注，文本补全模型是一种基于语言模型的机器学习模型，根据上下文的语境和语言规律，自动推断出最有可能的下一个文本补全。

| 输入 | 输出 |
| -------- | ------- |
| 一条文本内容 | 一条文本内容 |

### 聊天模型 (Chat Models)

聊天模型是语言模型的一种变体。聊天模型使用语言模型，并提供基于"聊天消息"的接口。

| 输入 | 输出 |
| -------- | ------- |
| 一组聊天消息 | 一条聊天消息 |

`聊天消息` 除了消息内容文本，还会包含一些其他参数数据。这在后续的内容中会看到。

## Langchain与OpenAI模型

参考OpenAI [Model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility) 文档，gpt模型都归为了聊天模型，而davinci, curie, babbage, ada模型都归为了文本补全模型。

| ENDPOINT | MODEL NAME |
| -------- | ------- |
| /v1/chat/completions | gpt-4, gpt-4-0613, gpt-4-32k, gpt-4-32k-0613, gpt-3.5-turbo, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-16k-0613 |
| /v1/completions | (Legacy)	text-davinci-003, text-davinci-002, text-davinci-001, text-curie-001, text-babbage-001, text-ada-001, davinci, curie, babbage, ada |

Langchain提供接口集成不同的模型。为了便于切换模型，Langchain将不同模型抽象为相同的接口 `BaseLanguageModel`，并提供 `predict` 和 `predict_messages` 函数来调用模型。

当使用LLM时推荐使用predict函数，当使用聊天模型时推荐使用predict_messages函数。

## 示例代码

接下来我们来看看如何在Langchain中使用LLM和聊天模型。

[Models.ipynb](./Models.ipynb)
### 与LLM的交互

与LLM的交互，我们需要使用 `langchain.llms` 模块中的 `OpenAI` 类。

```python
from langchain.llms import OpenAI

import os
os.environ['OPENAI_API_KEY'] = '您的有效OpenAI API Key'

llm = OpenAI(model_name="text-davinci-003")
response = llm.predict("What is AI?")
print(response)
```

你应该能看到类似如下输出：

```shell
AI (Artificial Intelligence) is a branch of computer science that deals with creating intelligent machines that can think, reason, learn, and problem solve. AI systems are designed to mimic human behavior and can be used to automate tasks or provide insights into data. AI can be used in a variety of fields, such as healthcare, finance, robotics, and more.
```

### 与聊天模型的交互

与聊天模型的交互，我们需要使用 `langchain.chat_models` 模块中的 `ChatOpenAI` 类。

```python
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

import os
os.environ['OPENAI_API_KEY'] = '您的有效OpenAI API Key'

chat = ChatOpenAI(temperature=0)
response = chat.predict_messages([ 
  HumanMessage(content="What is AI?")
])
print(response)
```

你应该能看到类似如下输出：

```shell
content='AI, or Artificial Intelligence, refers to the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, problem-solving, perception, and language understanding. AI technology has the capability to drastically change and improve the way we work, live, and interact.' additional_kwargs={} example=False
```

通过以下代码我们查看一下 `response` 变量的类型：

```python
response.__class
```

可以看到，它是一个 `AIMessage` 类型的对象。

```shell
langchain.schema.messages.AIMessage
```

```shell

接下来我们使用 `SystemMessage` 指令来指定模型的行为。如下代码指定模型对AI一无所知，在回答AI相关问题时，回答“I don't know”。

```python
response = chat.predict_messages([
  SystemMessage(content="You are a chatbot that knows nothing about AI. When you are asked about AI, you must say 'I don\'t know'"),
  HumanMessage(content="What is deep learning?")
])
print(response)
```

你应该能看到类似如下输出：

```shell
content="I don't know." additional_kwargs={} example=False
```

#### 3个消息类

Langchain框架提供了三个消息类，分别是 `AIMessage`、`HumanMessage` 和 `SystemMessage`。它们对应了OpenAI聊天模型API支持的不同角色 `assistant`、`user` 和 `system`。请参考 [OpenAI API文档 - Chat - Role](https://platform.openai.com/docs/api-reference/chat/create#chat/create-role)。

| Langchain类 | OpenAI角色 | 作用 |
| -------- | ------- | ------- |
| AIMessage | assistant | 模型回答的消息 |
| HumanMessage | user | 用户向模型的请求或提问 |
| SystemMessage | system | 系统指令，用于指定模型的行为 |

## 总结

本节课程中，我们学习了模型的基本概念，LLM与聊天模型的差异，并基于 `Langchain` 实现了分别与OpenAI LLM和聊天模型的交互。

要注意，虽然是聊天，但是当前我们所实现的交互并没有记忆能力，也就是说，模型并不会记住之前的对话内容。在后续的内容中，我们将学习如何实现记忆能力。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/get_started/introduction.html) 
2. [Models - Langchain](https://python.langchain.com/docs/modules/model_io/models/)