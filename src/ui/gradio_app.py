"""Minimal Gradio UI for FastAPI speaker verification backend."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

try:
    from src.ui.api_client import APIClientError, verify_with_api
except ModuleNotFoundError:  # Allows: python src/ui/gradio_app.py
    from api_client import APIClientError, verify_with_api  # type: ignore[no-redef]


def _decision_badge(decision: bool | None) -> str:
    if decision is True:
        title = "ACCEPTED"
        subtitle = "Same Speaker"
        bg = "#d1e7dd"
        fg = "#0f5132"
    elif decision is False:
        title = "REJECTED"
        subtitle = "Different Speaker"
        bg = "#f8d7da"
        fg = "#842029"
    else:
        title = "NO RESULT"
        subtitle = "Run verification to see decision"
        bg = "#e2e3e5"
        fg = "#41464b"

    return (
        f"<div style='padding:18px;border-radius:10px;background:{bg};color:{fg};"
        "text-align:center;border:1px solid rgba(0,0,0,0.08);'>"
        f"<div style='font-size:28px;font-weight:800;line-height:1.15;'>{title}</div>"
        f"<div style='font-size:18px;font-weight:600;margin-top:4px;'>{subtitle}</div>"
        "</div>"
    )


def _empty_result(status: str) -> tuple[str, str, str, float, float, int, str, str]:
    return (
        status,
        _decision_badge(None),
        "",
        0.0,
        0.0,
        0,
        "",
        "",
    )


def on_verify(
    enroll_audio_path: str | None,
    verification_audio_path: str | None,
    enhancement: bool,
) -> tuple[str, str, str, float, float, int, str, str]:
    """Validate inputs, call backend API, and format UI outputs."""
    if not enroll_audio_path:
        return _empty_result("Missing enrollment audio.")
    if not verification_audio_path:
        return _empty_result("Missing verification audio.")

    enroll_path = Path(enroll_audio_path)
    verification_path = Path(verification_audio_path)

    if not enroll_path.is_file():
        return _empty_result(f"Missing enrollment audio file: {enroll_path}")
    if not verification_path.is_file():
        return _empty_result(f"Missing verification audio file: {verification_path}")

    try:
        payload = verify_with_api(
            enroll_audio_path=enroll_path,
            test_audio_path=verification_path,
            enhancement=enhancement,
        )
    except (APIClientError, FileNotFoundError) as exc:
        error_text = str(exc)
        lowered = error_text.lower()
        if "cannot connect" in lowered or "timed out" in lowered:
            status = f"Backend unavailable. {error_text}"
        else:
            status = f"Verification failed. {error_text}"
        return _empty_result(status)

    status = "Backend connected. Verification completed."
    decision = bool(payload["decision"])
    return (
        status,
        _decision_badge(decision),
        str(payload["decision_label"]),
        float(payload["score"]),
        float(payload["threshold"]),
        int(payload["latency_ms"]),
        str(payload["model_name"]),
        str(payload["message"]),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Noise-Robust Speaker Verification") as demo:
        gr.Markdown(
            "# Noise-Robust Speaker Verification\n"
            "Upload or record an enrollment clip and a verification clip, then compare "
            "them using the local speaker verification backend."
        )

        with gr.Row():
            enroll_audio = gr.Audio(
                label="Enrollment Audio",
                sources=["upload", "microphone"],
                type="filepath",
            )
            verification_audio = gr.Audio(
                label="Verification Audio",
                sources=["upload", "microphone"],
                type="filepath",
            )

        enhancement = gr.Checkbox(label="Enhancement", value=False)
        verify_button = gr.Button("Verify", variant="primary")

        status = gr.Textbox(label="Status / Error", interactive=False)
        decision_badge = gr.HTML(value=_decision_badge(None), label="Decision")

        with gr.Row():
            decision_label = gr.Textbox(label="Decision", interactive=False)
            score = gr.Number(label="Score", interactive=False)
            threshold = gr.Number(label="Threshold", interactive=False)
            latency_ms = gr.Number(label="Latency (ms)", interactive=False)

        with gr.Row():
            model_name = gr.Textbox(label="Model Name", interactive=False)
            message = gr.Textbox(label="Message", interactive=False)

        verify_button.click(
            fn=on_verify,
            inputs=[enroll_audio, verification_audio, enhancement],
            outputs=[
                status,
                decision_badge,
                decision_label,
                score,
                threshold,
                latency_ms,
                model_name,
                message,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
