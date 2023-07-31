---
title: 06. 链
tags:
  - openai
  - llm
  - langchain
---

# WTF Langchain极简入门: 06. 链

最近在学习Langchain框架，顺手写一个“WTF Langchain极简入门”，供小白们使用（编程大佬可以另找教程）。本教程默认以下前提：
- 使用Python版本的[Langchain](https://github.com/hwchase17/langchain)
- LLM使用OpenAI的模型
- Langchain目前还处于快速发展阶段，版本迭代频繁，为避免示例代码失效，本教程统一使用版本 **0.0.235**

根据Langchain的[代码约定](https://github.com/hwchase17/langchain/blob/v0.0.235/pyproject.toml#L14C1-L14C24)，Python版本 ">=3.8.1,<4.0"。

推特：[@verysmallwoods](https://twitter.com/verysmallwoods)

所有代码和教程开源在github: [github.com/sugarforever/wtf-langchain](https://github.com/sugarforever/wtf-langchain)

-----

## 简介

单一的LLM对于简单的应用场景已经足够，但是更复杂的应用程序需要将LLM串联在一起，需要多LLM协同工作。

LangChain提出了 `链` 的概念，为这种“链式”应用程序提供了 **Chain** 接口。`Chain` 定义组件的调用序列，其中可以包括其他链。链大大简化复杂应用程序的实现，并使其模块化，这也使调试、维护和改进应用程序变得更容易。

## 最基础的链 LLMChain

作为极简教程，我们从最基础的概念，与组件开始。`LLMChain` 是 `LangChain` 中最基础的链。本课就从 `LLMChain` 开始，学习链的使用。

`LLMChain` 接受如下组件：
- LLM
- 提示词模版

`LLMChain` 返回LLM的回复。

在[第二讲](../02_Models/README.md)中我们学习了OpenAI LLM的使用。现在我们基于OpenAI LLM，利用 `LLMChain` 尝试构建自己第一个链。

1. 准备必要的组件

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0, openai_api_key="您的有效openai ai key")
prompt = PromptTemplate(
    input_variables=["color"],
    template="What is the hex code of color {color}?",
)
```

2. 基于组件创建 `LLMChain` 实例

我们要创建的链，基于提示词模版，提供基于颜色名字询问对应的16进制代码的能力。

```python
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
```

3. 基于链提问

现在我们利用创建的 `LLMChain` 实例提问。注意，提问中我们只需要提供第一步中创建的提示词模版变量的值。我们分别提问green，cyan，magento三种颜色的16进制代码。

```python
print(chain.run("green"))
print(chain.run("cyan"))
print(chain.run("magento"))
```

你应该期望如下输出：

```shell
The hex code of color green is #00FF00.
The hex code of color cyan is #00FFFF.
The hex code for the color Magento is #E13939.
```

## LangChainHub

[LangChainHub](https://github.com/hwchase17/langchain-hub) 收集并分享用于处理 `LangChain` 基本元素（提示词，链，和代理等）。

本讲，我们介绍 `LangChainHub` 中分享的链的使用。

### Hello World链

代码仓库：[https://github.com/hwchase17/langchain-hub/blob/master/chains/hello-world/](https://github.com/hwchase17/langchain-hub/blob/master/chains/hello-world/)

链定义：[https://github.com/hwchase17/langchain-hub/blob/master/chains/hello-world/chain.json](https://github.com/hwchase17/langchain-hub/blob/master/chains/hello-world/chain.json)

定义的原始内容：
```json
{
    "memory": null,
    "verbose": false,
    "prompt": {
        "input_variables": [
            "topic"
        ],
        "output_parser": null,
        "template": "Tell me a joke about {topic}:",
        "template_format": "f-string",
        "_type": "prompt"
    },
    "llm": {
        "model_name": "text-davinci-003",
        "temperature": 0.9,
        "max_tokens": 256,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "n": 1,
        "best_of": 1,
        "request_timeout": null,
        "logit_bias": {},
        "_type": "openai"
    },
    "output_key": "text",
    "_type": "llm_chain"
}
```

这条链，使用如下组件：
1. 提示词模版 - 请求LLM回答一个 `topic` 参数指定的话题的笑话
2. LLM - OpenAI的 `text-davince-003` 模型（包括模型相关参数的设置）

### 从LangChainHub加载链

本课以链[LLM-Math](https://github.com/hwchase17/langchain-hub/tree/master/chains/llm-math)为例，介绍如何从 `LangChainHub` 加载链并使用它。这是一个使用LLM和Python REPL来解决复杂数学问题的链。

#### 加载

使用 `load_chain` 函数从hub加载。

```python
from langchain.chains import load_chain
import os

os.environ['OPENAI_API_KEY'] = "您的有效openai api key"
chain = load_chain("lc://chains/llm-math/chain.json")
```

注：
1. OpenAI类允许通过参数 `openai_api_key` 指定API Key，也可以通过环境变量 `OPENAI_API_KEY` 自动加载。在本例中，load_chain函数完成加载，OpenAI的实例化由框架完成，因此在这里我们用了环境变量来指定API Key。
2. `load_chain` 函数的参数是hub中分享的链的json定义。参数格式：`lc://<链json文件在LangChainHub的相对路径>`。

#### 提问

现在我们可以基于这个链提问。

```python
chain.run("whats the area of a circle with radius 2?")
```

你应该期望如下输出：

```shell
> Entering new LLMMathChain chain...
whats the area of a circle with radius 2?
Answer: 12.566370614359172
> Finished chain.

Answer: 12.566370614359172
```

## 总结
本节课程中，我们学习了`LangChain` 提出的最重要的概念 - 链（`Chain`） ，介绍了如何使用链，并分享了如何利用开源社区的力量 - 从 `LangChainHub` 加载链，让LLM开发变得更加轻松。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/get_started/introduction.html) 
2. [LangChain Hub](https://github.com/hwchase17/langchain-hub)