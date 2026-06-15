# Interview Preparation — ML Engineer & GenAI Internship

## 20 Technical Questions with Detailed Answers

---

### 1. What is a Large Language Model (LLM) and how does it work?

**Answer:**
A Large Language Model is a neural network trained on massive text corpora to predict the next token in a sequence. Modern LLMs use the **Transformer** architecture, which relies on self-attention to model long-range dependencies in text. During pre-training the model learns statistical patterns across billions of tokens; during inference it autoregressively generates text token-by-token conditioned on the context (prompt). Scale — in parameters and data — is the primary driver of emergent capabilities.

---

### 2. Explain the Transformer architecture and its key components.

**Answer:**
The Transformer (Vaswani et al., 2017) replaced RNNs with pure attention. Core components:

| Component | Role |
|---|---|
| **Token Embedding** | Maps discrete tokens to dense vectors |
| **Positional Encoding** | Injects position information (sinusoidal or learned) |
| **Multi-Head Self-Attention** | Allows every token to attend to every other token |
| **Feed-Forward Network** | Two-layer MLP applied independently per position |
| **Layer Norm + Residuals** | Stabilise gradients; enable very deep stacks |

Decoders (GPT-family) use *causal masking* so token i can only attend to tokens ≤ i.

---

### 3. What is LoRA (Low-Rank Adaptation) and why is it efficient?

**Answer:**
LoRA freezes the pre-trained weight matrix **W** and injects two small matrices **A** (d×r) and **B** (r×k) so the effective weight update is **ΔW = BA**. Since rank r ≪ min(d,k), the number of trainable parameters drops by 10-10,000×. During inference you can merge the adapter back into W at zero cost. This makes fine-tuning tractable on consumer hardware without sacrificing performance compared to full fine-tuning, especially in the low-data regime.

---

### 4. What is PEFT and what adaptor types does it support?

**Answer:**
**PEFT (Parameter-Efficient Fine-Tuning)** is a HuggingFace library that provides unified APIs for adapting large models while keeping most parameters frozen. Supported methods include:

- **LoRA / QLoRA** — low-rank weight decomposition
- **Prefix Tuning** — prepend learned virtual tokens to each layer
- **Prompt Tuning** — soft prompt prepended only to the input
- **Adapter layers** — small bottleneck modules inserted between transformer blocks
- **IA³** — learn three scaling vectors per layer

LoRA is most widely used because it adds zero inference latency when merged.

---

### 5. What is TinyLlama and why was it chosen for this project?

**Answer:**
TinyLlama-1.1B-Chat is a 1.1-billion-parameter causal LM pre-trained on 3 trillion tokens using Llama-2 architecture. It was chosen because:
- It fits comfortably in CPU RAM (~2 GB in float32)
- The chat variant already understands instruction formats
- Its Llama-2 architecture makes it fully compatible with PEFT LoRA adapters targeting `q_proj` / `v_proj`
- It produces coherent text even on resource-limited environments like Replit

---

### 6. What is synthetic data generation and why is it valuable?

**Answer:**
Synthetic data generation produces labelled training examples programmatically — often using a stronger model — rather than hiring human annotators. Value:
1. **Cost**: generating 300 Q&A pairs with Mistral costs minutes vs. weeks of human labelling
2. **Scale**: easily generate thousands of diverse examples
3. **Privacy**: no real user data required
4. **Domain control**: target specific topics (e.g. one Wikipedia article)

Risks include distributional shift and hallucinated "facts" — mitigated here by grounding generation in actual Wikipedia text.

---

### 7. What is Ollama and how does it enable local LLM inference?

**Answer:**
Ollama is a cross-platform tool that packages GGUF-quantised LLMs into a single binary with a REST API (`localhost:11434`). It handles model download, quantisation selection, and GPU/CPU scheduling automatically. In this project Mistral runs via Ollama to generate synthetic Q&A pairs — no cloud API key or GPU required, and all data stays local.

---

### 8. What metrics are used to evaluate language model quality?

**Answer:**

| Metric | Measures | Range | Notes |
|---|---|---|---|
| **Exact Match (EM)** | Identical string after normalisation | 0–100% | Strict; good for factoid QA |
| **BLEU** | n-gram precision with brevity penalty | 0–100 | Standard in MT; prone to surface matching |
| **ROUGE-L** | LCS-based recall | 0–100 | Better for summarisation |
| **BERTScore** | Contextual embedding similarity | 0–100 | Model-based; captures paraphrase |

Using multiple complementary metrics avoids gaming any single one.

---

### 9. Explain the LoRA hyperparameters used in this project.

**Answer:**

| Param | Value | Explanation |
|---|---|---|
| `r` | 16 | Rank of the adaptation matrices — higher = more capacity |
| `lora_alpha` | 32 | Scaling factor: effective LR scale = α/r = 2.0 |
| `lora_dropout` | 0.05 | Light regularisation |
| `target_modules` | q_proj, v_proj | Attention projection weights — highest impact with lowest overhead |
| `bias` | none | Don't train bias terms — further reduces params |

Rank 16 is a sweet spot: significantly more expressive than r=4 while still being far smaller than full fine-tuning.

---

### 10. What is Prompt Engineering and how was it applied here?

**Answer:**
Prompt Engineering is the practice of crafting inputs that elicit desired outputs from a pre-trained LLM without weight updates. Applied here to drive the **synthetic data generator**:

- **Role priming**: framing Mistral as an educational question writer
- **Output format specification**: requiring strict JSON array output
- **Question type anchoring**: four distinct prompts (factual, reasoning, conceptual, definition) to maximise diversity
- **Length control**: truncating context to 1500 chars to stay within context window

Good prompts reduce hallucination and improve parse success rate.

---

### 11. What is quantisation and how does it relate to QLoRA?

**Answer:**
Quantisation reduces weight precision from float32/float16 → int8/int4, shrinking model size by 2-8× with minimal quality loss. **QLoRA** (Dettmers et al., 2023) combines 4-bit NF4 quantisation of the frozen base model with float16 LoRA adapters — enabling fine-tuning of 65B models on a single 48 GB GPU. This project uses float32 (CPU-safe) but the LoRA config is QLoRA-compatible by adding `BitsAndBytesConfig` on CUDA.

---

### 12. What is gradient accumulation and why is it used?

**Answer:**
Gradient accumulation simulates a large batch size on memory-constrained hardware. Instead of one backward pass per batch, gradients are accumulated over N steps before the optimiser update. Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps`. Here: 4 × 2 = 8 effective samples per update — balancing memory usage with stable gradient estimates.

---

### 13. How does BeautifulSoup parse Wikipedia articles?

**Answer:**
BeautifulSoup builds a DOM tree from HTML, allowing CSS-selector or tag-based navigation. The scraper:
1. Fetches the raw HTML with `requests`
2. Decomposes non-content tags (`<script>`, `<table>`, navboxes, references)
3. Iterates `<h2>/<h3>/<p>` inside `#mw-content-text`
4. Skips sections whose headings match an unwanted set (References, See Also …)
5. Joins remaining `<p>` text into a clean string

This beats simple regex parsing because HTML is not a regular language.

---

### 14. What is the HuggingFace `datasets` library and how is it used here?

**Answer:**
`datasets` provides fast, memory-mapped dataset loading with Arrow backend. Here it:
- Wraps the list of formatted training prompts in a `Dataset` object
- Provides `train_test_split()` for held-out evaluation
- Enables batched tokenisation via `.map()` with automatic multiprocessing
- Integrates directly with HuggingFace `Trainer` via its `__getitem__` protocol

---

### 15. What is BLEU score and what are its limitations?

**Answer:**
BLEU (Bilingual Evaluation Understudy) computes the geometric mean of n-gram precision scores (n=1..4) with a brevity penalty for short outputs. **Limitations**: it rewards surface-level overlap and penalises valid paraphrases; it is insensitive to meaning — a garbled sentence with shared words scores higher than a correct paraphrase. BERTScore addresses this by comparing contextual embeddings rather than exact token matches.

---

### 16. Describe the full data pipeline from Wikipedia to fine-tuned model.

**Answer:**
```
Wikipedia URL
  → requests + BeautifulSoup → raw text
  → clean (strip citations, whitespace) → clean_text.txt
  → word-based chunking (500w, 50w overlap) → chunks.json
  → Mistral via Ollama (4 prompt types) → synthetic_qa.json
  → HuggingFace Dataset + tokenise → tokenized splits
  → TinyLlama + LoRA (PEFT Trainer) → adapter weights
  → Merge → merged model
  → InferencePipeline → Gradio UI
```

---

### 17. How would you scale this pipeline to production?

**Answer:**
1. **Parallelise generation**: run Ollama workers in parallel with `asyncio` or `multiprocessing`
2. **Larger dataset**: loop over multiple Wikipedia articles or domains
3. **GPU training**: switch to QLoRA with BitsAndBytes on A100; use DeepSpeed ZeRO-3 for multi-GPU
4. **Experiment tracking**: integrate MLflow or W&B for metric logging per run
5. **Model registry**: version adapters with DVC or HuggingFace Hub
6. **Serving**: export to ONNX or vLLM for high-throughput inference
7. **Automated evaluation**: CI pipeline running ROUGE/BERTScore on a fixed test set after each adapter push

---

### 18. What is the difference between instruction tuning and domain adaptation?

**Answer:**
**Instruction tuning** teaches a model to follow diverse instructions in a chat format (e.g. FLAN-T5, Alpaca, TinyLlama-Chat). **Domain adaptation** shifts the model's knowledge distribution toward a specific domain (e.g. medical, legal, or a specific Wikipedia topic). This project blends both: the synthetic Q&A pairs are instruction-formatted (chat template) *and* domain-specific (transformer architecture topic).

---

### 19. What safety and quality checks does the generator apply?

**Answer:**
1. **JSON parsing with fallback**: regex extract → full-parse → discard
2. **Minimum length filter**: questions < 10 chars and answers < 5 chars discarded
3. **MD5 deduplication**: identical (normalised) questions hashed and skipped
4. **Retry logic**: up to 3 attempts with exponential back-off on Ollama failures
5. **Type labelling**: each pair carries its generation type for stratified analysis

---

### 20. How does the Gradio interface work technically?

**Answer:**
Gradio wraps Python functions as web endpoints using FastAPI and WebSockets under the hood. The `gr.Blocks` API provides layout primitives (Row, Column, Textbox, Slider) that bind to Python callables via `.click()` / `.submit()` event handlers. State is managed per-session by Gradio's built-in state mechanism. The `launch()` call starts a uvicorn server on the configured port, serving the UI as a single-page React app that communicates with the backend via WebSocket for streaming output.
