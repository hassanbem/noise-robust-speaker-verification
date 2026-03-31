"""Clean Gradio UI connected to local FastAPI backend."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

try:
    from src.ui.api_client import APIClientError, verify_with_api
except ModuleNotFoundError:  # Support `python src/ui/gradio_app.py`.
    from api_client import APIClientError, verify_with_api


def _neutral_badge(text: str = "No Decision Yet") -> str:
    return (
        "<div style='padding:16px;border-radius:12px;border:2px solid #9ca3af;"
        "background:#f3f4f6;color:#374151;font-weight:700;font-size:24px;"
        "text-align:center;'>"
        f"{text}"
        "</div>"
    )


def _decision_badge(decision: bool, decision_label: str) -> str:
    if decision:
        return (
            "<div style='padding:16px;border-radius:12px;border:2px solid #15803d;"
            "background:#dcfce7;color:#14532d;font-weight:800;font-size:26px;"
            "text-align:center;'>"
            "ACCEPTED / SAME SPEAKER"
            f"<div style='font-size:16px;font-weight:600;margin-top:6px;'>{decision_label}</div>"
            "</div>"
        )
    return (
        "<div style='padding:16px;border-radius:12px;border:2px solid #b91c1c;"
        "background:#fee2e2;color:#7f1d1d;font-weight:800;font-size:26px;"
        "text-align:center;'>"
        "REJECTED / DIFFERENT SPEAKER"
        f"<div style='font-size:16px;font-weight:600;margin-top:6px;'>{decision_label}</div>"
        "</div>"
    )


def _empty_outputs(
    status_message: str,
    message: str = "",
) -> tuple[str, str, float, float, int, str, str, str]:
    return (
        status_message,
        _neutral_badge(),
        "",
        0.0,
        0.0,
        0,
        "",
        message,
    )


def _validate_audio_input(audio_path: str | None, field_name: str) -> Path:
    if not audio_path:
        raise ValueError(f"{field_name} is required.")
    path = Path(audio_path)
    if not path.is_file():
        raise ValueError(f"{field_name} file does not exist: {path}")
    return path


def run_verification(
    enroll_audio: str | None,
    test_audio: str | None,
    enhancement: bool,
) -> tuple[str, str, float, float, int, str, str, str]:
    """Validate inputs, call backend, and format UI outputs."""
    try:
        enroll_path = _validate_audio_input(enroll_audio, "Enrollment audio")
        test_path = _validate_audio_input(test_audio, "Verification audio")
        response = verify_with_api(
            enroll_audio_path=enroll_path,
            test_audio_path=test_path,
            enhancement=enhancement,
        )
    except ValueError as exc:
        detail = str(exc)
        if "Enrollment audio" in detail:
            return _empty_outputs("Missing enrollment audio.", detail)
        if "Verification audio" in detail:
            return _empty_outputs("Missing verification audio.", detail)
        return _empty_outputs("Input validation error.", detail)
    except FileNotFoundError as exc:
        return _empty_outputs("Audio file not found.", str(exc))
    except APIClientError as exc:
        return _empty_outputs(
            "Backend unavailable. Start FastAPI on http://127.0.0.1:8000.",
            str(exc),
        )
    except Exception as exc:  # Defensive fallback for demo robustness.
        return _empty_outputs("Unexpected error during verification.", str(exc))

    status = "Backend connected. Verification completed."
    return (
        status,
        _decision_badge(response["decision"], response["decision_label"]),
        response["decision_label"],
        response["score"],
        response["threshold"],
        response["latency_ms"],
        response["model_name"],
        response["message"],
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Noise-Robust Speaker Verification Demo") as demo:
        gr.Markdown(
            "# Noise-Robust Speaker Verification\n"
            "Upload or record an enrollment clip and a verification clip, then compare them "
            "using the local speaker verification backend."
        )

        status_box = gr.Textbox(
            label="Status",
            value="Ready. Start backend at http://127.0.0.1:8000.",
            interactive=False,
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "Tip: microphone is selected first. Record your clip, then trim it before verification."
                )
                enroll_audio = gr.Audio(
                    label="Enrollment Audio",
                    sources=["microphone", "upload"],
                    type="filepath",
                    format="wav",
                    recording=False,
                    interactive=True,
                    editable=True,
                    waveform_options=gr.WaveformOptions(
                        sample_rate=16000,
                        trim_region_color="#f59e0b",
                        show_recording_waveform=True,
                    ),
                )
                test_audio = gr.Audio(
                    label="Verification Audio",
                    sources=["microphone", "upload"],
                    type="filepath",
                    format="wav",
                    recording=False,
                    interactive=True,
                    editable=True,
                    waveform_options=gr.WaveformOptions(
                        sample_rate=16000,
                        trim_region_color="#f59e0b",
                        show_recording_waveform=True,
                    ),
                )
                enhancement = gr.Checkbox(label="Enhancement", value=False)
                verify_button = gr.Button("Verify", variant="primary")

            with gr.Column():
                decision_badge = gr.HTML(value=_neutral_badge())
                with gr.Row():
                    score = gr.Number(label="Score", interactive=False)
                    threshold = gr.Number(label="Threshold", interactive=False)
                    latency_ms = gr.Number(label="Latency (ms)", interactive=False)
                decision_label = gr.Textbox(label="Decision Label", interactive=False)
                model_name = gr.Textbox(label="Model Name", interactive=False)
                message = gr.Textbox(label="Message", interactive=False)

        verify_button.click(
            fn=run_verification,
            inputs=[enroll_audio, test_audio, enhancement],
            outputs=[
                status_box,
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
    app.launch(server_name="127.0.0.1")
