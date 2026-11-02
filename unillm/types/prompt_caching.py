"""Type definitions for prompt caching configuration."""

from typing import List, Literal

# Components that can be targeted for prompt caching
PromptCacheParam = List[Literal["tools", "system", "messages"]]

# Every breakpoint a client places unless it is given its own list: Claude
# reads nothing from its prompt cache without one, and a breakpoint on a
# prefix that is never repeated costs only its write.
DEFAULT_PROMPT_CACHING: PromptCacheParam = ["tools", "system", "messages"]
