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
.card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center; }
.card-title { font-size: 0.9em; color: #64748b; font-weight: 600; margin-bottom: 4px; }
.card-value { font-size: 1.8em; color: #0f172a; font-weight: 700; }
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
        self.config = config
        self.cfg = config["gradio"]
        self.base_pipeline = base_pipeline
        self.ft_pipeline = ft_pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dashboard_html(self) -> str:
        """Load and format the Multi-Agent quality report as HTML."""
        import json
        
        report_path = Path(self.config.get("paths", {}).get("reports", "reports")) / "agent_quality_report.json"
        
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
        threshold = self.config.get("agents", {}).get("quality_threshold", 7.0)
        
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

    def launch(self, share: bool = False) -> None:
        """Build and start the Gradio server.

        Args:
            share: Whether to create a public Gradio share link.
        """
        import gradio as gr

        with gr.Blocks(css=CSS, title=TITLE, theme=gr.themes.Soft()) as demo:
            gr.Markdown(f"# {TITLE}")
            gr.Markdown(DESCRIPTION)

            with gr.Tabs():
                with gr.Tab("💬 Model Comparison"):
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

                with gr.Tab("📊 Multi-Agent Quality Dashboard"):
                    gr.Markdown("### 🧬 Data Quality & Orchestrator Telemetry")
                    dashboard_html = gr.HTML(value=self.get_dashboard_html())
                    refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")

                    refresh_btn.click(
                        fn=self.get_dashboard_html,
                        inputs=[],
                        outputs=dashboard_html,
                    )

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
