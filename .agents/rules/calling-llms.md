# Calling LLMs

- **Prompt caching on every call, as far as it goes.** A call that repeats
  an earlier call's opening reads it from the provider's prompt cache, at
  the longest-lived setting the provider offers, since a cache read costs a
  tenth of fresh input on Anthropic's models. How UniLLM places Claude's
  breakpoints, how a caller marks more and how to warm a prefix before
  fanning out are in the README's Prompt Caching section, which a change to
  that code keeps true. A call path that repeats a prefix and shows no
  cache reads in its responses' cached-token counts is a bug.
- **OpenAI only through OpenRouter.** Every OpenAI chat call goes through
  OpenRouter with `OPENROUTER_API_KEY`, as `openai/<model-id>@openrouter`.
  UniLLM registers no native OpenAI provider, so `<model-id>@openai` is not
  found: treat any surviving `@openai` string as a bug, and give env
  defaults and `.env.example` the `@openrouter` form. Chat and text
  generation never use `OPENAI_API_KEY` or a native `openai.OpenAI()`
  client.
- **Anthropic models keep OpenRouter's default routing.** Where an
  OpenRouter account carries its own Anthropic keys (bring-your-own-key),
  OpenRouter serves a Claude call from them first. A `provider.order` or
  `only` naming Amazon Bedrock, Google Vertex or any other host skips them
  and pays list price.
