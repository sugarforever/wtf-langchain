---
title: 09. 回调 (Callback)
tags:
  - openai
  - llm
  - langchain
---

# WTF Langchain极简入门: 09. 回调 (Callback)

最近在学习Langchain框架，顺手写一个“WTF Langchain极简入门”，供小白们使用（编程大佬可以另找教程）。本教程默认以下前提：
- 使用Python版本的[Langchain](https://github.com/hwchase17/langchain)
- LLM使用OpenAI的模型
- Langchain目前还处于快速发展阶段，版本迭代频繁，为避免示例代码失效，本教程统一使用版本 **0.0.235**

根据Langchain的[代码约定](https://github.com/hwchase17/langchain/blob/v0.0.235/pyproject.toml#L14C1-L14C24)，Python版本 ">=3.8.1,<4.0"。

推特：[@verysmallwoods](https://twitter.com/verysmallwoods)

所有代码和教程开源在github: [github.com/sugarforever/wtf-langchain](https://github.com/sugarforever/wtf-langchain)

-----

## 简介

`Callback` 是 `LangChain` 提供的回调机制，允许我们在 `LLM` 应用程序的各个阶段使用 `Hook`（钩子）。这对于记录日志、监控、流式传输等任务非常有用。这些任务的执行逻辑由回调处理器（`CallbackHandler`）定义。

在 `Python` 程序中， 回调处理器通过继承 `BaseCallbackHandler` 来实现。`BaseCallbackHandler` 接口对每一个可订阅的事件定义了一个回调函数。`BaseCallbackHandler` 的子类可以实现这些回调函数来处理事件。当事件触发时，`LangChain` 的回调管理器 `CallbackManager` 会调用相应的回调函数。

以下是 `BaseCallbackHandler` 的定义。请参考[源代码](https://github.com/langchain-ai/langchain/blob/v0.0.235/langchain/callbacks/base.py#L225)。

```python
class BaseCallbackHandler:
    """Base callback handler that can be used to handle callbacks from langchain."""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""

    def on_chat_model_start(
        self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any
    ) -> Any:
        """Run when Chat Model starts running."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
```

`LangChain` 内置支持了一系列回调处理器，我们也可以按需求自定义处理器，以实现特定的业务。

## 内置处理器

`StdOutCallbackHandler` 是 `LangChain` 所支持的最基本的处理器。它将所有的回调信息打印到标准输出。这对于调试非常有用。

`LangChain` 链的基类 `Chain` 提供了一个 `callbacks` 参数来指定要使用的回调处理器。请参考[`Chain源码`](https://github.com/langchain-ai/langchain/blob/v0.0.235/langchain/chains/base.py#L63)，其中代码片段为：

```python
class Chain(Serializable, ABC):
    """Abstract base class for creating structured sequences of calls to components.
    ...
    callbacks: Callbacks = Field(default=None, exclude=True)
    """Optional list of callback handlers (or callback manager). Defaults to None.
    Callback handlers are called throughout the lifecycle of a call to a chain,
    starting with on_chain_start, ending with on_chain_end or on_chain_error.
    Each custom chain can optionally call additional callback methods, see Callback docs
    for full details."""
```

用法如下：

```python
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

handler = StdOutCallbackHandler()
llm = OpenAI()
prompt = PromptTemplate.from_template("Who is {name}?")
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[handler])
chain.run(name="Super Mario")
```

你应该期望如下输出：

```shell
> Entering new LLMChain chain...
Prompt after formatting:
Who is Super Mario?

> Finished chain.

\n\nSuper Mario is the protagonist of the popular video game franchise of the same name created by Nintendo. He is a fictional character who stars in video games, television shows, comic books, and films. He is a plumber who is usually portrayed as a portly Italian-American, who is often accompanied by his brother Luigi. He is well known for his catchphrase "It\'s-a me, Mario!"
```

## 自定义处理器

我们可以通过继承 `BaseCallbackHandler` 来实现自定义的回调处理器。下面是一个简单的例子，`TimerHandler` 将跟踪 `Chain` 或 `LLM` 交互的起止时间，并统计每次交互的处理耗时。

```python
from langchain.callbacks.base import BaseCallbackHandler
import time

class TimerHandler(BaseCallbackHandler):

    def __init__(self) -> None:
        super().__init__()
        self.previous_ms = None
        self.durations = []

    def current_ms(self):
        return int(time.time() * 1000 + time.perf_counter() % 1 * 1000)

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        self.previous_ms = self.current_ms()

    def on_chain_end(self, outputs, **kwargs) -> None:
        if self.previous_ms:
          duration = self.current_ms() - self.previous_ms
          self.durations.append(duration)

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.previous_ms = self.current_ms()

    def on_llm_end(self, response, **kwargs) -> None:
        if self.previous_ms:
          duration = self.current_ms() - self.previous_ms
          self.durations.append(duration)

llm = OpenAI()
timerHandler = TimerHandler()
prompt = PromptTemplate.from_template("What is the HEX code of color {color_name}?")
chain = LLMChain(llm=llm, prompt=prompt, callbacks=[timerHandler])
response = chain.run(color_name="blue")
print(response)
response = chain.run(color_name="purple")
print(response)

timerHandler.durations
```

你应该期望如下输出：

```shell
The HEX code for blue is #0000FF.
The HEX code of the color purple is #800080.
[1589, 1097]
```

## 回调处理器的适用场景

通过 `LLMChain` 的构造函数参数设置 `callbacks` 仅仅是众多适用场景之一。接下来我们简明地列出其他使用场景和示例代码。

对于 `Model`，`Agent`， `Tool`，以及 `Chain` 都可以通过以下方式设置回调处理器：
### 1. 构造函数参数 `callbacks` 设置

关于 `Chain`，以 `LLMChain` 为例，请参考本讲上一部分内容。注意在 `Chain` 上的回调器监听的是 `chain` 相关的事件，因此回调器的如下函数会被调用：
- on_chain_start
- on_chain_end
- on_chain_error

`Agent`， `Tool`，以及 `Chain` 上的回调器会分别被调用相应的回调函数。

下面分享关于 `Model` 与 `callbacks` 的使用示例：

```python
timerHandler = TimerHandler()
llm = OpenAI(callbacks=[timerHandler])
response = llm.predict("What is the HEX code of color BLACK?")
print(response)

timerHandler.durations
```

你应该期望看到类似如下的输出：

```shell
['What is the HEX code of color BLACK?']
generations=[[Generation(text='\n\nThe hex code of black is #000000.', generation_info={'finish_reason': 'stop', 'logprobs': None})]] llm_output={'token_usage': {'prompt_tokens': 10, 'total_tokens': 21, 'completion_tokens': 11}, 'model_name': 'text-davinci-003'} run=None


The hex code of black is #000000.

[1223]
```

### 2. 通过运行时的函数调用

`Model`，`Agent`， `Tool`，以及 `Chain` 的请求执行函数都接受 `callbacks` 参数，比如 `LLMChain` 的 `run` 函数，`OpenAI` 的 `predict` 函数，等都能接受 `callbacks` 参数，在运行时指定回调处理器。

以 `OpenAI` 模型类为例：

```python
timerHandler = TimerHandler()
llm = OpenAI()
response = llm.predict("What is the HEX code of color BLACK?", callbacks=[timerHandler])
print(response)

timerHandler.durations
```

你应该同样期望如下输出：

```shell
['What is the HEX code of color BLACK?']
generations=[[Generation(text='\n\nThe hex code of black is #000000.', generation_info={'finish_reason': 'stop', 'logprobs': None})]] llm_output={'token_usage': {'prompt_tokens': 10, 'total_tokens': 21, 'completion_tokens': 11}, 'model_name': 'text-davinci-003'} run=None

The hex code of black is #000000.

[1138]
```

关于 `Agent`，`Tool` 等的使用，请参考官方文档API。

## 总结

本节课程中，我们学习了什么是 `Callback` 回调，如何使用回调处理器，以及在哪些场景下可以接入回调处理器。下一讲，我们将一起完成一个完整的应用案例，来巩固本系列课程的知识点。

本节课程的完整示例代码，请参考 [09_Callbacks.ipynb](./09_Callbacks.ipynb)。

### 相关文档资料链接：
1. [Python Langchain官方文档](https://python.langchain.com/docs/get_started/introduction.html) 