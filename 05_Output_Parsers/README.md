---
title: 05. 输出解析器
tags:
  - openai
  - llm
  - langchain
---

# WTF Langchain极简入门: 05. 输出解析器

最近在学习Langchain框架，顺手写一个“WTF Langchain极简入门”，供小白们使用（编程大佬可以另找教程）。本教程默认以下前提：
- 使用Python版本的[Langchain](https://github.com/hwchase17/langchain)
- LLM使用OpenAI的模型
- Langchain目前还处于快速发展阶段，版本迭代频繁，为避免示例代码失效，本教程统一使用版本 **0.0.235**

根据Langchain的[代码约定](https://github.com/hwchase17/langchain/blob/v0.0.235/pyproject.toml#L14C1-L14C24)，Python版本 ">=3.8.1,<4.0"。

推特：[@verysmallwoods](https://twitter.com/verysmallwoods)

所有代码和教程开源在github: [github.com/sugarforever/wtf-langchain](https://github.com/sugarforever/wtf-langchain)

-----

## 简介

LLM的输出为文本，但在程序中除了显示文本，可能希望获得更结构化的数据。这就是输出解析器（Output Parsers）的用武之地。

`LangChain` 为输出解析器提供了基础类 `BaseOutputParser`。不同的输出解析器都继承自该类。它们需要实现以下两个函数：
- **get_format_instructions**：返回指令指定LLM的输出该如何格式化，该函数在实现类中必须重写。基类中的函数实现如下：
```python
def get_format_instructions(self) -> str:
    """Instructions on how the LLM output should be formatted."""
    raise NotImplementedError
```
- **parse**：解析LLM的输出文本为特定的结构。函数签名如下：
```python
def parse(self, text: str) -> T
```

`BaseOutputParser` 还提供了如下函数供重载：
**parse_with_prompt**：基于提示词上下文解析LLM的输出文本为特定结构。该函数在基类中的实现为：
```python
def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:
    """Parse the output of an LLM call with the input prompt for context."""
    return self.parse(completion)
```

## LangChain支持的输出解析器

LangChain框架提供了一系列解析器实现来满足应用在不同功能场景中的需求。它们包括且不局限于如下解析器：
- List parser
- Datetime parser
- Enum parser
- Auto-fixing parser
- Pydantic parser
- Retry parser
- Structured output parser

本讲介绍具有代表性的两款解析器的使用。

### List Parser

List Parser将逗号分隔的文本解析为列表。

```python
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
output_parser.parse("black, yellow, red, green, white, blue")
```

你应该能看到如下输出：

```shell
['black', 'yellow', 'red', 'green', 'white', 'blue']
```

### Structured Output Parser

当我们想要类似JSON数据结构，包含多个字段时，可以使用这个输出解析器。该解析器可以生成指令帮助LLM返回结构化数据文本，同时完成文本到结构化数据的解析工作。示例代码如下：

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI

# 定义响应的结构(JSON)，两个字段 answer和source。
response_schemas = [
    ResponseSchema(name="answer", description="answer to the user's question"),
    ResponseSchema(name="source", description="source referred to answer the user's question, should be a website.")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 获取响应格式化的指令
format_instructions = output_parser.get_format_instructions()

# partial_variables允许在代码中预填充提示此模版的部分变量。这类似于接口，抽象类之间的关系
prompt = PromptTemplate(
    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

model = OpenAI(temperature=0)
response = prompt.format_prompt(question="what's the capital of France?")
output = model(response.to_string())
output_parser.parse(output)
```

你应该期望能看到如下输出：
```shell
{
    'answer': 'Paris',
    'source': 'https://www.worldatlas.com/articles/what-is-the-capital-of-france.html'
}
```

注，关于示例代码中引用的 `partial_variables`，请参考[Partial - Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/partial)。

## 总结
本节课程中，我们学习了什么是 `输出解析器` ，LangChain所支持的常见解析器，以及如何使用常见的两款解析器 `List Parser` 和 `Structured Output Parser`。随着LangChain的发展，应该会有更多解析器出现。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/get_started/introduction.html) 
2. [Partial - Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/partial)