"""
launch_ui.py — Launch the Gradio web interface with iframe embedding support.

Wraps Gradio in a FastAPI app with middleware that removes X-Frame-Options
so the preview pane (iframe) can display it correctly.

Usage:
  python launch_ui.py
  python launch_ui.py --base-only   # skip fine-tuned model (faster)
  python launch_ui.py --share       # public Gradio share link
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.utils.helpers import load_config, setup_logging, ensure_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch the Gradio comparison UI")
    p.add_argument("--base-only", action="store_true", help="Only load the base model")
    p.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    p.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)
    setup_logging(config["logging"]["file"])
    logger = logging.getLogger("launch_ui")

    port = int(os.environ.get("PORT", config["gradio"].get("port", 8000)))
    config["gradio"]["port"] = port
    logger.info("Starting Gradio app on port %d …", port)

    import gradio as gr
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    # ── Shared model state (loaded in background thread) ─────────────
    state: dict = {"base": None, "ft": None, "loading": True, "error": None}

    def _load_models() -> None:
        from src.inference.pipeline import InferencePipeline
        try:
            logger.info("Background: loading base model …")
            base = InferencePipeline(config, use_fine_tuned=False)
            base.load()
            state["base"] = base

            if args.base_only:
                state["ft"] = base
                logger.info("--base-only: using base model for both columns.")
            else:
                logger.info("Background: loading fine-tuned model …")
                ft = InferencePipeline(config, use_fine_tuned=True)
                ft.load()
                state["ft"] = ft

            state["loading"] = False
            logger.info("✓ Both models ready.")
        except Exception as exc:
            state["error"] = str(exc)
            state["loading"] = False
            logger.exception("Model loading failed: %s", exc)

    loader_thread = threading.Thread(target=_load_models, daemon=True)
    loader_thread.start()

    # ── Gradio UI ─────────────────────────────────────────────────────
    CSS = """
    .model-box { border-radius: 12px; padding: 16px; }
    .base-box  { border: 1px solid #667eea60;
                 background: linear-gradient(135deg,#667eea12,#764ba212); }
    .ft-box    { border: 1px solid #f093fb60;
                 background: linear-gradient(135deg,#f093fb12,#f5576c12); }
    .metrics   { font-size:0.84em; color:#666; margin-top:6px; }
    footer { display: none !important; }
    """

    EXAMPLES = [
        "What is the transformer architecture in deep learning?",
        "How does self-attention mechanism work?",
        "What is the difference between encoder and decoder?",
        "Explain multi-head attention.",
        "What are feed-forward layers in transformers?",
        "How does positional encoding work?",
        "What is BERT and how was it trained?",
        "Explain the concept of tokenization in NLP.",
    ]

    def respond(question: str, temp: float, max_tok: int) -> tuple:
        if not question.strip():
            return ("Please enter a question.", "", "", "")
        if state["loading"]:
            return (
                "⏳ Models are still loading — please wait a moment and try again.\n\n"
                "Click **Refresh Status** above to check progress.",
                "", "", "",
            )
        if state["error"]:
            return (f"❌ Model loading failed: {state['error']}", "", "", "")

        base_p, ft_p = state["base"], state["ft"]
        base_p.cfg_inf["temperature"] = temp
        base_p.cfg_inf["max_new_tokens"] = int(max_tok)
        ft_p.cfg_inf["temperature"] = temp
        ft_p.cfg_inf["max_new_tokens"] = int(max_tok)

        base_res = base_p.generate(question)
        ft_res   = ft_p.generate(question)

        base_meta = f"⏱ **{base_res['latency_s']}s** | 📊 **{base_res['token_count']}** tokens"
        ft_meta   = f"⏱ **{ft_res['latency_s']}s** | 📊 **{ft_res['token_count']}** tokens"
        return base_res["response"], ft_res["response"], base_meta, ft_meta

    def check_status() -> str:
        if state["loading"]:
            return (
                "⏳ **Loading models in the background…**  "
                "First run downloads ~2 GB of weights (cached after that). "
                "Click **Refresh Status** to check."
            )
        if state["error"]:
            return f"❌ **Error:** {state['error']}"
        return "✅ **Models ready!** Enter a question below and click Generate."

    def clear_all() -> tuple:
        return "", "", "", "", ""

    # In Gradio 6.0 css/theme moved to launch(); with mount_gradio_app we
    # pass them here — Gradio still honours them, it's just a warning.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        demo_ctx = gr.Blocks(title="TinyLlama Fine-Tuning Demo",
                             css=CSS, theme=gr.themes.Soft())

    with demo_ctx as demo:
        gr.Markdown("# 🤖 TinyLlama Fine-Tuning Demo")
        gr.Markdown(
            "Compare **Base TinyLlama-1.1B** vs **LoRA Fine-Tuned TinyLlama** "
            "trained on Wikipedia synthetic Q&A pairs generated via Mistral + Ollama."
        )

        status_box  = gr.Markdown(value=check_status())
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        refresh_btn.click(fn=check_status, outputs=status_box)

        with gr.Row():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about the topic the model was trained on…",
                lines=3, scale=4,
            )
            with gr.Column(scale=1):
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                max_tokens  = gr.Slider(64, 512, value=256, step=32, label="Max New Tokens")
                submit_btn  = gr.Button("Generate ▶", variant="primary")
                clear_btn   = gr.Button("Clear")

        with gr.Row():
            with gr.Column(elem_classes="model-box base-box"):
                gr.Markdown("### 🔵 Base TinyLlama-1.1B")
                base_output  = gr.Textbox(label="Response", lines=10, interactive=False)
                base_metrics = gr.Markdown(elem_classes="metrics")

            with gr.Column(elem_classes="model-box ft-box"):
                gr.Markdown("### 🟣 Fine-Tuned TinyLlama (LoRA)")
                ft_output  = gr.Textbox(label="Response", lines=10, interactive=False)
                ft_metrics = gr.Markdown(elem_classes="metrics")

        gr.Examples(examples=EXAMPLES, inputs=question_input, label="Example Questions")

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

    # ── FastAPI wrapper — strips X-Frame-Options so iframes work ──────
    fapp = FastAPI()

    @fapp.middleware("http")
    async def allow_iframe(request: Request, call_next):
        response: Response = await call_next(request)
        # Remove the header that blocks iframe embedding
        # MutableHeaders uses del, not .pop()
        if "x-frame-options" in response.headers:
            del response.headers["x-frame-options"]
        # Allow embedding from any origin
        response.headers["content-security-policy"] = "frame-ancestors *"
        return response

    fapp = gr.mount_gradio_app(fapp, demo, path="/")

    logger.info("Serving on http://0.0.0.0:%d", port)
    uvicorn.run(fapp, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
