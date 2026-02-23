"""Type definitions for prompt caching configuration."""

from typing import List, Literal

# Components that can be targeted for prompt caching
PromptCacheParam = List[Literal["tools", "system", "messages"]]
