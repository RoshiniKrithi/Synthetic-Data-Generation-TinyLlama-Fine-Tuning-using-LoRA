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

logger = logging.getLogger("launch_ui")


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

    port = int(os.environ.get("PORT", config["gradio"].get("port", 8000)))
    config["gradio"]["port"] = port
    logger.info("Starting Gradio app on port %d …", port)

    import gradio as gr
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import Response

    # ── Demo model (small, fits in free-tier RAM) ─────────────────────
    demo_model = config.get("gradio", {}).get("demo_model", "gpt2")
    logger.info("UI demo model: %s", demo_model)

    # ── Shared model state (loaded in background thread) ─────────────
    state: dict = {"base": None, "ft": None, "loading": True, "error": None,
                   "demo_model": demo_model}

    def _load_models() -> None:
        from src.inference.pipeline import InferencePipeline
        try:
            logger.info("Background: loading demo model (%s) …", demo_model)
            base = InferencePipeline(config, use_fine_tuned=False,
                                     model_name_override=demo_model)
            base.load()
            state["base"] = base

            adapter_dir = ROOT / config.get("paths", {}).get("models_adapter", "models/adapter")
            adapter_exists = adapter_dir.exists() and any(adapter_dir.iterdir())

            if args.base_only or not adapter_exists:
                # Reuse the same model object — avoids double RAM usage
                state["ft"] = base
                if not adapter_exists:
                    logger.info(
                        "No LoRA adapter found at %s — "
                        "using demo model for both columns (run Phase 4 to train).",
                        adapter_dir,
                    )
                else:
                    logger.info("--base-only: using demo model for both columns.")
            else:
                logger.info("Background: loading fine-tuned model …")
                ft = InferencePipeline(config, use_fine_tuned=True,
                                       model_name_override=demo_model)
                ft.load()
                state["ft"] = ft

            state["loading"] = False
            logger.info("[OK] Both models ready.")
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
    .card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center; }
    .card-title { font-size: 0.9em; color: #64748b; font-weight: 600; margin-bottom: 4px; }
    .card-value { font-size: 1.8em; color: #0f172a; font-weight: 700; }
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
        adapter_dir = ROOT / config.get("paths", {}).get("models_adapter", "models/adapter")
        adapter_exists = adapter_dir.exists() and any(adapter_dir.iterdir())
        if not adapter_exists:
            return (
                "✅ **Base model ready!** (both columns use base TinyLlama — "
                "run `python run_all.py --phases 4` in the shell to train the LoRA adapter, "
                "then restart this app to compare)"
            )
        return "✅ **Models ready!** Enter a question below and click Generate."

    def get_dashboard_html() -> str:
        import json
        
        report_path = Path(config.get("paths", {}).get("reports", "reports")) / "agent_quality_report.json"
        
        if not report_path.exists():
            return """
            <div style='padding: 32px; border: 2px dashed #cbd5e1; border-radius: 12px; text-align: center; background: #f8fafc; margin-top: 16px;'>
                <h3 style='margin: 0 0 8px 0; color: #64748b; font-size: 1.2em;'>No Multi-Agent Quality Report Found</h3>
                <p style='margin: 0; color: #94a3b8; font-size: 0.95em;'>
                    Please run the generation pipeline with the Multi-Agent system enabled to generate data quality statistics:
                </p>
                <code style='display: block; background: #0f172a; color: #f8fafc; padding: 12px; border-radius: 6px; margin: 16px auto 0 auto; max-width: 500px; text-align: left; font-family: monospace;'>python run_all.py --phases 1 2 3 --use-agents</code>
            </div>
            """
        
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Error loading quality report: {e}")
            return f"<div style='color: red; padding: 16px;'>Error loading quality report: {e}</div>"
            
        sum_data = report.get("summary", {})
        qual_data = report.get("quality", {})
        ref_data = report.get("refinement", {})
        dist = report.get("score_distribution", {})
        dim_avgs = qual_data.get("dimension_averages", {})
        threshold = config.get("agents", {}).get("quality_threshold", 7.0)
        
        # Build HTML summary grid
        html = f"""
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 16px 0 24px 0;'>
            <div class='card' style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center;'>
                <div class='card-title' style='font-size: 0.9em; color: #64748b; font-weight: 600; margin-bottom: 4px;'>Total Generated</div>
                <div class='card-value' style='font-size: 1.8em; color: #0f172a; font-weight: 700;'>{sum_data.get("total_generated", 0)}</div>
            </div>
            <div class='card' style='background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 8px; padding: 16px; text-align: center;'>
                <div class='card-title' style='font-size: 0.9em; color: #047857; font-weight: 600; margin-bottom: 4px;'>Accepted (Rate)</div>
                <div class='card-value' style='font-size: 1.8em; color: #065f46; font-weight: 700;'>{sum_data.get("total_accepted", 0)} ({sum_data.get("acceptance_rate_pct", 0)}%)</div>
            </div>
            <div class='card' style='background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 16px; text-align: center;'>
                <div class='card-title' style='font-size: 0.9em; color: #b91c1c; font-weight: 600; margin-bottom: 4px;'>Rejected / Duplicates</div>
                <div class='card-value' style='font-size: 1.8em; color: #991b1b; font-weight: 700;'>{sum_data.get("total_rejected", 0)} / {sum_data.get("total_duplicates", 0)}</div>
            </div>
            <div class='card' style='background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; text-align: center;'>
                <div class='card-title' style='font-size: 0.9em; color: #1d4ed8; font-weight: 600; margin-bottom: 4px;'>Average Overall Score</div>
                <div class='card-value' style='font-size: 1.8em; color: #1e40af; font-weight: 700;'>{qual_data.get("average_overall_score", 0)} / 10</div>
            </div>
        </div>
        
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px;'>
            <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px;'>
                <h3 style='margin: 0 0 12px 0; color: #334155; font-size: 1.1em; border-bottom: 2px solid #f1f5f9; padding-bottom: 6px;'>Dimension Breakdown</h3>
                <table style='width: 100%; border-collapse: collapse; text-align: left;'>
                    <thead>
                        <tr style='border-bottom: 1px solid #cbd5e1;'>
                            <th style='padding: 8px 0; color: #64748b; font-weight: 600;'>Dimension</th>
                            <th style='padding: 8px 0; text-align: right; color: #64748b; font-weight: 600;'>Average Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style='border-bottom: 1px solid #f1f5f9;'>
                            <td style='padding: 8px 0; color: #334155;'>🎯 Relevance</td>
                            <td style='padding: 8px 0; text-align: right; font-weight: 600; color: #0f172a;'>{dim_avgs.get("relevance", 0)} / 10</td>
                        </tr>
                        <tr style='border-bottom: 1px solid #f1f5f9;'>
                            <td style='padding: 8px 0; color: #334155;'>✨ Clarity</td>
                            <td style='padding: 8px 0; text-align: right; font-weight: 600; color: #0f172a;'>{dim_avgs.get("clarity", 0)} / 10</td>
                        </tr>
                        <tr style='border-bottom: 1px solid #f1f5f9;'>
                            <td style='padding: 8px 0; color: #334155;'>📝 Answer Quality</td>
                            <td style='padding: 8px 0; text-align: right; font-weight: 600; color: #0f172a;'>{dim_avgs.get("answer_quality", 0)} / 10</td>
                        </tr>
                        <tr style='border-bottom: 1px solid #f1f5f9;'>
                            <td style='padding: 8px 0; color: #334155;'>🎓 Educational Value</td>
                            <td style='padding: 8px 0; text-align: right; font-weight: 600; color: #0f172a;'>{dim_avgs.get("educational_value", 0)} / 10</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px;'>
                <h3 style='margin: 0 0 12px 0; color: #334155; font-size: 1.1em; border-bottom: 2px solid #f1f5f9; padding-bottom: 6px;'>Self-Correction & Refinement</h3>
                <table style='width: 100%; border-collapse: collapse; text-align: left;'>
                    <thead>
                        <tr style='border-bottom: 1px solid #cbd5e1;'>
                            <th style='padding: 8px 0; color: #64748b; font-weight: 600;'>Metric</th>
                            <th style='padding: 8px 0; text-align: right; color: #64748b; font-weight: 600;'>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style='border-bottom: 1px solid #f1f5f9;'>
                            <td style='padding: 8px 0; color: #334155;'>🔄 Pairs Requiring Refinement</td>
                            <td style='padding: 8px 0; text-align: right; font-weight: 600; color: #0f172a;'>{ref_data.get("pairs_refined", 0)}</td>
                        </tr>
                        <tr style='border-bottom: 1px solid #f1f5f9;'>
                            <td style='padding: 8px 0; color: #334155;'>📈 Avg Refinement Steps</td>
                            <td style='padding: 8px 0; text-align: right; font-weight: 600; color: #0f172a;'>{ref_data.get("avg_refinement_iterations", 0)} / {ref_data.get("max_refinement_iterations", 0)} max</td>
                        </tr>
                        <tr style='border-bottom: 1px solid #f1f5f9;'>
                            <td style='padding: 8px 0; color: #334155;'>⚙️ Score Gate Threshold</td>
                            <td style='padding: 8px 0; text-align: right; font-weight: 600; color: #0f172a;'>{threshold} / 10</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div style='background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px;'>
            <h3 style='margin: 0 0 12px 0; color: #334155; font-size: 1.1em; border-bottom: 2px solid #f1f5f9; padding-bottom: 6px;'>Score Distribution</h3>
            <div style='display: flex; gap: 8px; align-items: flex-end; height: 160px; padding: 16px 0; border-bottom: 1px solid #cbd5e1;'>
        """
        
        # Score distribution visualization
        max_val = max(dist.values()) if dist else 1
        for bin_label, count in dist.items():
            pct = (count / max_val) * 120 if max_val > 0 else 0
            html += f"""
                <div style='flex: 1; display: flex; flex-direction: column; align-items: center;'>
                    <div style='color: #64748b; font-size: 0.8em; margin-bottom: 4px;'>{count}</div>
                    <div style='width: 60%; background: linear-gradient(to top, #3b82f6, #60a5fa); height: {pct:.1f}px; border-radius: 4px 4px 0 0; min-height: 4px;'></div>
                    <div style='color: #475569; font-size: 0.85em; font-weight: 600; margin-top: 8px;'>{bin_label}</div>
                </div>
            """
            
        html += """
            </div>
        </div>
        """
        return html

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
            f"Interactive demo using **{demo_model}** (lightweight preview model). "
            "The full pipeline trains **TinyLlama-1.1B + LoRA** on Wikipedia synthetic Q&A — "
            "run `python run_all.py --phases 4` to train, then restart to compare."
        )

        with gr.Tabs():
            with gr.Tab("💬 Model Comparison"):
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

            with gr.Tab("📊 Multi-Agent Quality Dashboard"):
                gr.Markdown("### 🧬 Data Quality & Orchestrator Telemetry")
                dashboard_html = gr.HTML(value=get_dashboard_html())
                refresh_dashboard_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")

                refresh_dashboard_btn.click(
                    fn=get_dashboard_html,
                    inputs=[],
                    outputs=dashboard_html,
                )

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
