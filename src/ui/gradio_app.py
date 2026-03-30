"""Gradio demo app for Student 3.

This UI uses a fake backend contract first and can later be wired to a real
FastAPI endpoint.
"""

from __future__ import annotations

from pathlib import Path
import json

import gradio as gr

from src.ui.mock_backend import fake_verify

ENROLLMENTS: dict[str, str] = {}


def _speaker_choices() -> list[str]:
    return sorted(ENROLLMENTS.keys())


def enroll_speaker(speaker_name: str, enrollment_audio: str | None):
    """Store enrollment audio path in local memory for demo flow."""
    speaker_name = (speaker_name or "").strip()
    if not speaker_name:
        return "Please enter a speaker name.", gr.update(choices=_speaker_choices())
    if not enrollment_audio:
        return "Please record or upload enrollment audio.", gr.update(choices=_speaker_choices())

    audio_path = Path(enrollment_audio)
    if not audio_path.exists():
        return "Enrollment audio file does not exist.", gr.update(choices=_speaker_choices())

    ENROLLMENTS[speaker_name] = str(audio_path)
    status = f"Enrollment saved for '{speaker_name}' ({audio_path.name})."
    return status, gr.update(choices=_speaker_choices(), value=speaker_name)


def refresh_speakers():
    return gr.update(choices=_speaker_choices())


def verify_speaker(
    speaker_name: str,
    test_audio: str | None,
    noise_type: str,
    snr_db: int,
    enhancement: bool,
    threshold_mode: str,
):
    """Run mock verification and expose contract fields in the UI."""
    if not speaker_name:
        return (
            "<div style='padding:10px;border-radius:8px;background:#fff3cd;color:#664d03;'>Choose an enrolled speaker first.</div>",
            0.0,
            0.0,
            "N/A",
            0,
            "none",
            0,
            "{}",
        )
    if not test_audio:
        return (
            "<div style='padding:10px;border-radius:8px;background:#fff3cd;color:#664d03;'>Provide a test audio sample.</div>",
            0.0,
            0.0,
            "N/A",
            0,
            "none",
            0,
            "{}",
        )
    if speaker_name not in ENROLLMENTS:
        return (
            "<div style='padding:10px;border-radius:8px;background:#f8d7da;color:#842029;'>Selected speaker has no enrollment yet.</div>",
            0.0,
            0.0,
            "N/A",
            0,
            "none",
            0,
            "{}",
        )

    response = fake_verify(
        enroll_path=ENROLLMENTS[speaker_name],
        test_path=test_audio,
        threshold_mode=threshold_mode,
        enhancement=enhancement,
    )

    if response["decision"]:
        card = "<div style='padding:10px;border-radius:8px;background:#d1e7dd;color:#0f5132;font-weight:700;'>ACCEPT: same speaker</div>"
    else:
        card = "<div style='padding:10px;border-radius:8px;background:#f8d7da;color:#842029;font-weight:700;'>REJECT: different speaker</div>"

    return (
        card,
        response["score"],
        response["threshold"],
        response["decision_label"],
        response["latency_ms"],
        noise_type,
        snr_db,
        json.dumps(response, indent=2),
    )


def build_about_markdown() -> str:
    lines = [
        "### About This Demo",
        "- Model (contract): SpeechBrain ECAPA-TDNN",
        "- Input expectation: mono 16 kHz audio",
        "- Verification score: cosine similarity",
        "",
        "### Metric quick guide",
        "- EER: equal error rate where FAR equals FRR.",
        "- FAR: false acceptance rate (impostor accepted).",
        "- FRR: false rejection rate (genuine speaker rejected).",
    ]

    ablation_file = Path("results/ablation/noise_ablation.csv")
    if ablation_file.exists():
        lines += ["", f"Noise ablation table found: `{ablation_file}`"]
    else:
        lines += ["", "Noise ablation table not found yet (expected later from Student 1)."]

    return "\n".join(lines)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Noise-Robust Speaker Verification Demo") as demo:
        gr.Markdown(
            "# Speaker Verification Demo (Student 3)\n"
            "Uses a fake backend function that follows `docs/api_contract.md`."
        )

        with gr.Tab("Enroll"):
            speaker_name = gr.Textbox(label="Speaker name", placeholder="e.g. speaker_01")
            enrollment_audio = gr.Audio(
                label="Enrollment audio (upload or record)",
                sources=["upload", "microphone"],
                type="filepath",
            )
            enroll_btn = gr.Button("Save Enrollment", variant="primary")
            enroll_status = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("Verify"):
            with gr.Row():
                verify_speaker_name = gr.Dropdown(
                    label="Enrolled speaker",
                    choices=_speaker_choices(),
                )
                refresh_btn = gr.Button("Refresh speakers")

            test_audio = gr.Audio(
                label="Test audio (upload or record)",
                sources=["upload", "microphone"],
                type="filepath",
            )

            with gr.Row():
                noise_type = gr.Dropdown(
                    label="Noise mode",
                    choices=["clean", "music", "speech", "noise", "street", "call_center"],
                    value="clean",
                )
                snr_db = gr.Slider(label="SNR (dB)", minimum=-5, maximum=15, value=5, step=5)

            with gr.Row():
                threshold_mode = gr.Radio(
                    label="Threshold mode",
                    choices=["fixed", "eer", "far_1"],
                    value="fixed",
                )
                enhancement = gr.Checkbox(label="Enhancement enabled", value=False)

            verify_btn = gr.Button("Verify", variant="primary")
            decision_card = gr.HTML()

            with gr.Row():
                score = gr.Number(label="Similarity score")
                threshold = gr.Number(label="Threshold")
                decision_label = gr.Textbox(label="Decision label")
                latency_ms = gr.Number(label="Latency (ms)")

            with gr.Row():
                out_noise_type = gr.Textbox(label="Noise type", interactive=False)
                out_snr_db = gr.Number(label="SNR (dB)", interactive=False)

            contract_json = gr.Code(label="Backend JSON (contract)", language="json")

        with gr.Tab("Results / About"):
            gr.Markdown(build_about_markdown())
            figures_dir = Path("results/figures")
            if figures_dir.exists():
                figure_files = sorted(list(figures_dir.glob("*.png")))[:4]
                if figure_files:
                    for fig in figure_files:
                        gr.Image(value=str(fig), label=fig.name)
                else:
                    gr.Markdown("No PNG figures found yet in `results/figures`.")
            else:
                gr.Markdown("`results/figures` is not available yet.")

        enroll_btn.click(
            fn=enroll_speaker,
            inputs=[speaker_name, enrollment_audio],
            outputs=[enroll_status, verify_speaker_name],
        )

        refresh_btn.click(fn=refresh_speakers, inputs=None, outputs=verify_speaker_name)

        verify_btn.click(
            fn=verify_speaker,
            inputs=[verify_speaker_name, test_audio, noise_type, snr_db, enhancement, threshold_mode],
            outputs=[
                decision_card,
                score,
                threshold,
                decision_label,
                latency_ms,
                out_noise_type,
                out_snr_db,
                contract_json,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch()
