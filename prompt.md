# MoE Vision-Language Connector

## Architecture

```
                    f_drop                                    z_text
                      │                                         │
          ┌───┬───┬───┬───┬───┐                     ┌───┬───┬───┬───┬───┐
          │ v │ v │ v │ v │ v │  visual tokens       │ t │ t │ t │ t │ t │  text tokens
          └───┴───┴─┬─┴───┴───┘                     └───┴───┴───┴─┬─┴───┘
                    │                                              │
                    ▼                                              │
              ┌──────────┐                                         │
              │  Memory  │                                         │
              └────┬─────┘                                         │
                   │ z_drop                                        │
                   │                                               │
                   ▼                                               ▼
        ┌──────────────────────────────────────────────────────────────┐
        │                         Router                               │
        │                                                              │
        │    G_guiding ──► σ (sigmoid)      G_router ──► top-k softmax │
        │    (text-conditioned              (expert dispatch            │
        │     relevance gate)                weights)                   │
        │                                                              │
        └────────┬──────────┬──────────────────┬──────────┬────────────┘
                 │          │                  │          │
                 ▼          ▼                  ▼          ▼
          ┌──────────┐┌──────────┐     ┌──────────┐┌──────────┐
          │ Expert 1 ││ Expert 2 │     │ Expert 3 ││ Expert 4 │
          │  (MLP)   ││  (MLP)   │     │  (MLP)   ││  (MLP)   │
          └────┬─────┘└────┬─────┘     └────┬─────┘└────┬─────┘
               │           │                │           │
               └─────┬─────┴────────┬───────┴─────┬─────┘
                     ▼              ▼             ▼
               ┌───┬───┬───┬───┐
               │ v'│ v'│ v'│ v'│  projected visual tokens
               └───┴───┴───┴───┘
```

## Data Flow

1. **Visual tokens** pass through `f_drop` (feature dropout) and are compressed by a **Memory** module into `z_drop`.
2. **Text embeddings** (`z_text`) are extracted from the language model.
3. Both `z_drop` and `z_text` enter the **Router**, which contains two parallel gates:
   - **G_guiding + σ (sigmoid):** Produces per-token relevance scores conditioned on text, allowing independent retention or suppression of each visual token.
   - **G_router + top-k softmax:** Computes expert dispatch weights, selecting the top-k experts per token for processing.
4. Selected **Expert MLPs** process the routed visual tokens in parallel.
5. Expert outputs are combined to produce **projected visual tokens** (`v'`) aligned to the language model's embedding space.