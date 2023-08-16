---
title: 04. 提示词
tags:
  - openai
  - llm
  - langchain
---

# WTF Langchain极简入门: 04. 提示词

最近在学习Langchain框架，顺手写一个“WTF Langchain极简入门”，供小白们使用（编程大佬可以另找教程）。本教程默认以下前提：
- 使用Python版本的[Langchain](https://github.com/hwchase17/langchain)
- LLM使用OpenAI的模型
- Langchain目前还处于快速发展阶段，版本迭代频繁，为避免示例代码失效，本教程统一使用版本 **0.0.235**

根据Langchain的[代码约定](https://github.com/hwchase17/langchain/blob/v0.0.235/pyproject.toml#L14C1-L14C24)，Python版本 ">=3.8.1,<4.0"。

推特：[@verysmallwoods](https://twitter.com/verysmallwoods)

所有代码和教程开源在github: [github.com/sugarforever/wtf-langchain](https://github.com/sugarforever/wtf-langchain)

-----

## 什么是提示词？

提示词（`Prompt`）是指向模型提供的输入。这个输入通常由多个元素构成。`LangChain` 提供了一系列的类和函数，简化构建和处理提示词的过程。
- 提示词模板（Prompt Template）：对提示词参数化，提高代码的重用性。
- 示例选择器（Example Selector）：动态选择要包含在提示词中的示例

## 提示词模板

提示词模板提供了可重用提示词的机制。用户通过传递一组参数给模板，实例化图一个提示词。一个提示模板可以包含：
1. 对语言模型的指令
2. 一组少样本示例，以帮助语言模型生成更好的回复
3. 向语言模型提出的问题

一个简单的例子：

```python
from langchain import PromptTemplate
template = """
你精通多种语言，是专业的翻译官。你负责{src_lang}到{dst_lang}的翻译工作。
"""

prompt = PromptTemplate.from_template(template)
prompt.format(src_lang="英文", dst_lang="中文")
```

### 创建模板

`PromptTemplate` 类是 `LangChain` 提供的模版基础类，它接收两个参数：
1. `input_variables` - 输入变量
2. `template` - 模版

模版中通过 `{}` 符号来引用输入变量，比如 `PromptTemplate(input_variables=["name"], template="My name is {name}.")`。

模版的实例化通过模板类实例的 `format`函数实现。例子如下：

```python
multiple_input_prompt = PromptTemplate(
    input_variables=["color", "animal"], 
    template="A {color} {animal} ."
)
multiple_input_prompt.format(color="black", animal="bear")
```

#### 聊天提示词模板

聊天模型，比如 `OpenAI` 的GPT模型，接受一系列聊天消息作为输入，每条消息都与一个角色相关联。这个消息列表通常以一定格式串联，构成模型的输入，也就是提示词。

例如，在OpenAI [Chat Completion API](https://platform.openai.com/docs/api-reference/chat/create)中，聊天消息可以与assistant、human或system角色相关联。

为此，LangChain提供了一系列模板，以便更轻松地构建和处理提示词。建议在与聊天模型交互时优先选择使用这些与聊天相关的模板，而不是基础的PromptTemplate，以充分利用框架的优势，提高开发效率。`SystemMessagePromptTemplate`, `AIMessagePromptTemplate`, `HumanMessagePromptTemplate` 是分别用于创建不同角色提示词的模板。

我们来看一个综合示例：

```python
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

system_template="You are a professional translator that translates {src_lang} to {dst_lang}."
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template="{user_input}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chat_prompt.format_prompt(
    src_lang="English",
    dst_lang="Chinese", 
    user_input="Did you eat in this morning?"
).to_messages()
```

你应该能看到如下输出：
```shell
[SystemMessage(content='You are a professional translator that translates English to Chinese.', additional_kwargs={}),
 HumanMessage(content='Did you eat in this morning?', additional_kwargs={}, example=False)]
```

#### 样本选择器

在LLM应用开发中，可能需要从大量样本数据中，选择部分数据包含在提示词中。样本选择器（Example Selector）正是满足该需求的组件，它也通常与少样本提示词配合使用。`LangChain` 提供了样本选择器的基础接口类 `BaseExampleSelector`，每个选择器类必须实现的函数为 `select_examples`。`LangChain` 实现了若干基于不用应用场景或算法的选择器：
- LengthBasedExampleSelector 
- MaxMarginalRelevanceExampleSelector
- NGramOverlapExampleSelector
- SemanticSimilarityExampleSelector

本讲以基于长度的样本选择器（输入越长，选择的样本越少；输入越短，选择的样本越多）`LengthBasedExampleSelector` 为例，演示用法。

```python
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector


# These are a lot of examples of a pretend task of creating antonyms.
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
    {"input": "energetic", "output": "lethargic"},
    {"input": "sunny", "output": "gloomy"},
    {"input": "windy", "output": "calm"},
]
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    # 可选的样本数据
    examples=examples, 
    # 提示词模版
    example_prompt=example_prompt, 
    # 格式化的样本数据的最大长度，通过get_text_length函数来衡量
    max_length=25
)
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:", 
    input_variables=["adjective"],
)

# 输入量极小，因此所有样本数据都会被选中
print(dynamic_prompt.format(adjective="big"))
```

你应该能看到如下输出：
```shell
Give the antonym of every input

Input: happy
Output: sad

Input: tall
Output: short

Input: energetic
Output: lethargic

Input: sunny
Output: gloomy

Input: windy
Output: calm

Input: big
Output:
```

注，选择器实例化时，我们没有改变 `get_text_length` 函数实现，其默认实现为：

```python
    def _get_length_based(text: str) -> int:
        return len(re.split("\n| ", text))

    ......

    get_text_length: Callable[[str], int] = _get_length_based
    """Function to measure prompt length. Defaults to word count."""
```

## 总结
本节课程中，我们简要介绍了LLM中的重要概念 `提示词` 并学习了如何使用 `Langchain` 的重要组件 `提示词模板`。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/get_started/introduction.html) 