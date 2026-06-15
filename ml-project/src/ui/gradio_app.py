"""
Phase 7 — Gradio Web Interface

Professional side-by-side comparison of base vs fine-tuned TinyLlama.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = [
    "What is the transformer architecture in deep learning?",
    "How does attention mechanism work in neural networks?",
    "What is the difference between encoder and decoder in transformers?",
    "Explain the concept of self-attention.",
    "What are the key components of BERT?",
    "How does positional encoding work?",
    "What is multi-head attention?",
    "Explain the role of feed-forward layers in transformers.",
]

CSS = """
.container { max-width: 1200px; margin: auto; }
.model-box { border-radius: 12px; padding: 16px; }
.base-box { background: linear-gradient(135deg, #667eea20, #764ba220); border: 1px solid #667eea40; }
.ft-box   { background: linear-gradient(135deg, #f093fb20, #f5576c20); border: 1px solid #f093fb40; }
.metrics  { font-size: 0.85em; color: #666; margin-top: 8px; }
.title    { text-align: center; margin-bottom: 8px; }
"""

TITLE = "🤖 TinyLlama Fine-Tuning Demo"
DESCRIPTION = (
    "Compare **Base TinyLlama-1.1B** vs **LoRA Fine-Tuned TinyLlama** on your own questions.\n\n"
    "The fine-tuned model was trained on synthetic Q&A pairs generated from Wikipedia via Mistral + Ollama."
)


class GradioApp:
    """Build and launch the Gradio comparison interface.

    Args:
        config: Full project config dict.
        base_pipeline: Loaded InferencePipeline (base model).
        ft_pipeline: Loaded InferencePipeline (fine-tuned model).
    """

    def __init__(self, config: dict, base_pipeline, ft_pipeline) -> None:
        self.cfg = config["gradio"]
        self.base_pipeline = base_pipeline
        self.ft_pipeline = ft_pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def launch(self, share: bool = False) -> None:
        """Build and start the Gradio server.

        Args:
            share: Whether to create a public Gradio share link.
        """
        import gradio as gr

        with gr.Blocks(css=CSS, title=TITLE, theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"# {TITLE}")
            gr.Markdown(DESCRIPTION)

            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the topic the model was trained on …",
                    lines=3,
                    scale=4,
                )
                with gr.Column(scale=1):
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                    max_tokens = gr.Slider(64, 512, value=256, step=32, label="Max New Tokens")
                    submit_btn = gr.Button("Generate ▶", variant="primary")
                    clear_btn = gr.Button("Clear")

            with gr.Row():
                with gr.Column(elem_classes="model-box base-box"):
                    gr.Markdown("### 🔵 Base TinyLlama-1.1B", elem_classes="title")
                    base_output = gr.Textbox(label="Response", lines=10, interactive=False)
                    base_metrics = gr.Markdown(elem_classes="metrics")

                with gr.Column(elem_classes="model-box ft-box"):
                    gr.Markdown("### 🔴 Fine-Tuned TinyLlama (LoRA)", elem_classes="title")
                    ft_output = gr.Textbox(label="Response", lines=10, interactive=False)
                    ft_metrics = gr.Markdown(elem_classes="metrics")

            with gr.Row():
                gr.Examples(
                    examples=EXAMPLE_QUESTIONS,
                    inputs=question_input,
                    label="Example Questions",
                )

            with gr.Accordion("📥 Download Results", open=False):
                download_btn = gr.Button("Save last comparison to file")
                download_file = gr.File(label="Download")

            # State to hold last results for download
            last_result: list[dict] = [{}]

            def respond(question: str, temp: float, max_tok: int):
                """Run both pipelines and return formatted outputs."""
                if not question.strip():
                    return ("Please enter a question.", "", "", "")

                self.base_pipeline.cfg_inf["temperature"] = temp
                self.base_pipeline.cfg_inf["max_new_tokens"] = int(max_tok)
                self.ft_pipeline.cfg_inf["temperature"] = temp
                self.ft_pipeline.cfg_inf["max_new_tokens"] = int(max_tok)

                base_res = self.base_pipeline.generate(question)
                ft_res = self.ft_pipeline.generate(question)

                last_result[0] = {
                    "question": question,
                    "base": base_res,
                    "fine_tuned": ft_res,
                }

                base_meta = (
                    f"⏱ **{base_res['latency_s']}s** | "
                    f"📊 **{base_res['token_count']} tokens** generated"
                )
                ft_meta = (
                    f"⏱ **{ft_res['latency_s']}s** | "
                    f"📊 **{ft_res['token_count']} tokens** generated"
                )
                return base_res["response"], ft_res["response"], base_meta, ft_meta

            def save_results():
                """Write last results to a JSON file and return its path."""
                import json, tempfile
                if not last_result[0]:
                    return None
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                )
                json.dump(last_result[0], tmp, indent=2, ensure_ascii=False)
                tmp.close()
                return tmp.name

            def clear_all():
                return "", "", "", "", ""

            submit_btn.click(
                fn=respond,
                inputs=[question_input, temperature, max_tokens],
                outputs=[base_output, ft_output, base_metrics, ft_metrics],
            )
            question_input.submit(
                fn=respond,
                inputs=[question_input, temperature, max_tokens],
                outputs=[base_output, ft_output, base_metrics, ft_metrics],
            )
            clear_btn.click(
                fn=clear_all,
                outputs=[question_input, base_output, ft_output, base_metrics, ft_metrics],
            )
            download_btn.click(fn=save_results, outputs=download_file)

        port = int(os.environ.get("PORT", self.cfg.get("port", 7860)))
        demo.launch(
            server_name=self.cfg.get("host", "0.0.0.0"),
            server_port=port,
            share=share,
            show_error=True,
        )
