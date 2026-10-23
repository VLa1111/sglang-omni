# SPDX-License-Identifier: Apache-2.0
"""API tests for the native qwen3_asr_server, run against the real model.

Needs NATIVE_RUNTIME_BIN (the server's directory) and CI_DATA_ROOT (a provisioned root).
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import subprocess
import time
import uuid
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from websockets.sync.client import connect

RUNTIME_BIN = os.environ.get("NATIVE_RUNTIME_BIN")
DATA_ROOT = os.environ.get("CI_DATA_ROOT")
pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
MODEL_REPO = "mlx-community/Qwen3-ASR-0.6B-4bit"


def model_directory() -> Path:
    return Path(DATA_ROOT) / "models" / MODEL_REPO.replace("/", "_")


def clip(clip_id: str) -> bytes:
    return (Path(DATA_ROOT) / "corpus" / "v1" / "clips" / f"{clip_id}.wav").read_bytes()


def pcm16(clip_id: str) -> bytes:
    with wave.open(io.BytesIO(clip(clip_id))) as reader:
        return reader.readframes(reader.getnframes())


class Server:
    def __init__(
        self,
        binary: str = "qwen3_asr_server",
        model_kind: str = "qwen3_asr",
        directory: Path | None = None,
        flags: tuple[str, ...] = (),
    ) -> None:
        self.process = subprocess.Popen(
            [
                str(Path(RUNTIME_BIN) / binary),
                "--supervised",
                "--model-kind",
                model_kind,
                "--model-directory",
                str(directory or model_directory()),
                *flags,
            ],  # fmt: skip
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.ready = json.loads(self.process.stdout.readline())
        self.port = self.ready.get("port")

    def request(
        self, method: str, path: str, body: bytes = b"", headers: dict | None = None
    ):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()

    def post_form(self, fields: dict[str, str], wav: bytes | None):
        boundary = uuid.uuid4().hex
        body = b""
        for name, value in fields.items():
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        if wav is not None:
            body += (
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                    "Content-Type: audio/wav\r\n\r\n"
                ).encode()
                + wav
                + b"\r\n"
            )
        else:
            pass
        body += f"--{boundary}--\r\n".encode()
        return self.request(
            "POST",
            "/v1/audio/transcriptions",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.write('{"command": "shutdown"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=10)
        else:
            pass


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    running = Server()
    yield running
    running.stop()


def sse_events(body: bytes) -> list:
    return [
        (
            line[len("data: ") :]
            if line == "data: [DONE]"
            else json.loads(line[len("data: ") :])
        )
        for line in body.decode().splitlines()
        if line.startswith("data: ")
    ]


def test_ready_event_names_a_loopback_endpoint(server: Server) -> None:
    assert server.ready["event"] == "ready"
    assert server.ready["host"] == "127.0.0.1"
    assert server.ready["server_pid"] == server.process.pid
    assert server.ready["model_name"].startswith("voxt-qwen3_asr-")
    status, body = server.request("GET", "/health")
    assert (status, json.loads(body)) == (
        200,
        {"status": "healthy", "running": True, "request_states": {}},
    )
    status, body = server.request("GET", "/v1/models")
    assert json.loads(body)["data"] == [
        {"id": server.ready["model_name"], "object": "model"}
    ]


def test_voxt_final_request_streams_text_and_generation_metadata(
    server: Server,
) -> None:
    status, body = server.post_form(
        {
            "model": server.ready["model_name"],
            "stream": "true",
            "language": "English",
            "max_new_tokens": "1024",
            "stop_at_end_of_text": "true",
            "stop_on_token_loop": "true",
            "include_generation_metadata": "true",
            "audio_layout": "voxt_swift",
        },
        clip("0006_en_short"),
    )
    assert status == 200
    events = sse_events(body)
    assert events[-1] == "[DONE]"
    assert events[0] == {
        "type": "transcript.text.done",
        "text": "Surely you are not thinking of going off there.",
        "generation_metadata": {
            "generated_token_count": 11,
            "language": "English",
            "finish_reason": "stop",
        },
    }


def test_plain_request_returns_json_text(server: Server) -> None:
    status, body = server.post_form({"language": "en"}, clip("0006_en_short"))
    assert (status, json.loads(body)) == (
        200,
        {"text": "Surely you are not thinking of going off there."},
    )


@pytest.mark.parametrize(
    ("fields", "wav"),
    [
        ({}, None),
        ({}, b"not audio"),
        ({"include_generation_metadata": "true"}, "wav"),
        ({"audio_layout": "sideways"}, "wav"),
        ({"max_new_tokens": "many"}, "wav"),
        ({"max_new_tokens": "-1"}, "wav"),
    ],
)
def test_invalid_requests_are_rejected(server: Server, fields: dict, wav) -> None:
    status, body = server.post_form(
        fields, clip("0006_en_short") if wav == "wav" else wav
    )
    assert status == 400
    detail = json.loads(body)["detail"]
    assert all(name in detail for name in fields)


def long_wav(seconds: int) -> bytes:
    """A clip repeated to about seconds long, so a full decode takes several seconds."""
    pcm = pcm16("0344_en_long")
    repeated = pcm * (seconds * 32000 // len(pcm) + 1)
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(repeated[: seconds * 32000])
    return out.getvalue()


def test_a_disconnected_stream_stops_its_decode(server: Server) -> None:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="stream"\r\n\r\ntrue\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="max_new_tokens"\r\n\r\n4096\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
        ).encode()
        + long_wav(300)
        + f"\r\n--{boundary}--\r\n".encode()
    )
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=30)
    connection.request(
        "POST",
        "/v1/audio/transcriptions",
        body,
        {"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    response = connection.getresponse()
    assert response.status == 200
    # Note (Jiaxin Deng): cancellation is checked between generated tokens, so
    # give the request time to get past encoding and prefill.
    time.sleep(4.0)
    assert json.loads(server.request("GET", "/health")[1])["request_states"] == {
        "running": 1
    }
    closed_at = time.monotonic()
    # Note (Jiaxin Deng): http.client keeps the socket open while the response
    # object lives.
    response.close()
    connection.close()
    while json.loads(server.request("GET", "/health")[1])["request_states"] != {}:
        assert (
            time.monotonic() - closed_at < 1.5
        ), "the decode kept running after its client left"
        time.sleep(0.05)


def test_realtime_manual_session(server: Server) -> None:
    pcm = pcm16("0006_en_short")
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": {"type": "server_vad"}},
                }
            )
        )
        assert json.loads(socket.recv())["error"]["code"] == "unsupported_session"
        socket.send("{not json")
        assert json.loads(socket.recv())["error"]["code"] == "invalid_json"
        socket.send(json.dumps({"type": "input_audio_buffer.append", "audio": "***"}))
        assert json.loads(socket.recv())["error"]["code"] == "invalid_audio"
        socket.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": "AAAAAA"})
        )
        assert json.loads(socket.recv())["error"]["code"] == "invalid_audio"
        socket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"turn_detection": None, "language": "en"},
                }
            )
        )
        assert json.loads(socket.recv())["type"] == "transcription_session.updated"
        for start in range(0, len(pcm), 3200):
            socket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm[start : start + 3200]).decode(),
                    }
                )
            )
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(json.dumps({"type": "transcription.done"}))
        events = []
        while not events or events[-1]["type"] != "transcription.completed":
            events.append(json.loads(socket.recv()))
    finals = [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]
    assert [event["text"] for event in finals] == [events[-1]["text"]]
    assert events[-1]["text"].startswith("Surely you are not thinking")
    indexes = [event["event_index"] for event in events]
    assert indexes == sorted(indexes)


def receive_until(socket, event_type: str, timeout: float = 120.0) -> list[dict]:
    events = []
    while not events or events[-1]["type"] != event_type:
        events.append(json.loads(socket.recv(timeout=timeout)))
    return events


def append_pcm(
    socket, pcm: bytes, chunk: int = 3200, pace_seconds: float = 0.0
) -> None:
    for start in range(0, len(pcm), chunk):
        socket.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[start : start + chunk]).decode(),
                }
            )
        )
        if pace_seconds > 0.0:
            time.sleep(pace_seconds)
        else:
            pass


def update_session(socket, language: str = "en") -> list[dict]:
    socket.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {"turn_detection": None, "language": language},
            }
        )
    )
    return receive_until(socket, "transcription_session.updated")


def final_segments(events: list[dict]) -> list[dict]:
    return [
        event
        for event in events
        if event["type"] == "transcription.segment" and event["is_final"]
    ]


def joined_text(segments: list[dict]) -> str:
    texts = (segment["text"].strip() for segment in segments)
    return " ".join(text for text in texts if text)


def test_unsupported_realtime_events_keep_the_session_open(server: Server) -> None:
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        update_session(socket)
        # Note (VLa1111): clear and server-side items are not part of the manual
        # turn contract, so they are reported and the session keeps running.
        for message_type in ("input_audio_buffer.clear", "conversation.item.create"):
            socket.send(json.dumps({"type": message_type}))
            event = json.loads(socket.recv())
            assert event["type"] == "error"
            assert event["error"]["code"] == "invalid_event"
            assert event["error"]["message"] == "Unknown event type."
        assert [event["type"] for event in update_session(socket)] == [
            "transcription_session.updated"
        ]


def test_realtime_previews_are_replaced_before_the_final(server: Server) -> None:
    pcm = pcm16("0006_en_short")
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        update_session(socket)
        append_pcm(socket, pcm, pace_seconds=0.05)
        deadline = time.monotonic() + 60.0
        previews = []
        while len(previews) < 3 and time.monotonic() < deadline:
            try:
                event = json.loads(
                    socket.recv(timeout=max(deadline - time.monotonic(), 0.1))
                )
            except TimeoutError:
                break
            assert event["type"] == "transcription.segment"
            assert event["is_final"] is False
            previews.append(event)
        assert len(previews) >= 3, previews
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(json.dumps({"type": "transcription.done"}))
        events = receive_until(socket, "transcription.completed")
    finals = final_segments(events)
    assert [segment["text"] for segment in finals] == [events[-1]["text"]]
    # Note (VLa1111): every preview belongs to the one open segment, and the
    # text really is replaced as more audio arrives.
    assert len({preview["segment_id"] for preview in previews}) == 1
    assert len({preview["text"] for preview in previews}) >= 2
    first_final = next(
        index
        for index, event in enumerate(events)
        if event["type"] == "transcription.segment" and event["is_final"]
    )
    assert all(
        event["is_final"]
        for event in events[first_final:]
        if event["type"] == "transcription.segment"
    )
    assert events[-1]["text"].startswith("Surely")


def test_realtime_final_text_survives_a_wrong_language_preview(server: Server) -> None:
    # Note (VLa1111): asking for English makes the early previews of a Chinese
    # clip translate it; the final must still transcribe the Chinese audio.
    pcm = pcm16("0153_zh_short")
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        update_session(socket, language="English")
        append_pcm(socket, pcm, pace_seconds=0.1)
        deadline = time.monotonic() + 30.0
        previews = []
        while len(previews) < 3 and time.monotonic() < deadline:
            try:
                event = json.loads(
                    socket.recv(timeout=max(deadline - time.monotonic(), 0.1))
                )
            except TimeoutError:
                break
            if event["type"] == "transcription.segment" and not event["is_final"]:
                previews.append(event)
            else:
                pass
        assert len(previews) >= 3, previews
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(json.dumps({"type": "transcription.done"}))
        events = receive_until(socket, "transcription.completed")
    final = events[-1]["text"]
    assert any("㐀" <= character <= "鿿" for character in final), final


def test_repeated_commits_emit_one_final_per_segment(server: Server) -> None:
    pcm = pcm16("0006_en_short")
    chunk = 4800
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        update_session(socket)
        finals = []
        for cycle in range(10):
            if cycle % 2 == 0:
                append_pcm(socket, pcm[cycle // 2 * chunk : cycle // 2 * chunk + chunk])
            else:
                pass
            socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            # Note (VLa1111): the update response doubles as a barrier, because
            # the commit is handled before the session update on this socket.
            cycle_finals = final_segments(update_session(socket))
            if cycle % 2 == 0:
                assert len(cycle_finals) == 1
            else:
                assert cycle_finals == []
            finals.extend(cycle_finals)
        socket.send(json.dumps({"type": "transcription.done"}))
        completed = receive_until(socket, "transcription.completed")[-1]
    assert len(finals) == 5
    assert [segment["segment_id"] for segment in finals] == [0, 1, 2, 3, 4]
    assert completed["text"] == joined_text(finals)


def test_a_closed_realtime_socket_cancels_its_decode(server: Server) -> None:
    pcm = (pcm16("0344_en_long") * 2)[: 20 * 32000]
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        update_session(socket)
        append_pcm(socket, pcm)
        deadline = time.monotonic() + 30.0
        while json.loads(server.request("GET", "/health")[1])["request_states"] != {
            "running": 1
        }:
            assert time.monotonic() < deadline, "no realtime decode started"
            time.sleep(0.05)
        closed_at = time.monotonic()
        socket.close()
        while json.loads(server.request("GET", "/health")[1])["request_states"] != {}:
            assert (
                time.monotonic() - closed_at < 1.5
            ), "the realtime decode kept running after its socket closed"
            time.sleep(0.05)
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        update_session(socket)
        append_pcm(socket, pcm16("0006_en_short"))
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(json.dumps({"type": "transcription.done"}))
        events = receive_until(socket, "transcription.completed")
    assert events[-1]["text"].startswith("Surely")


def test_segments_split_at_the_configured_boundary() -> None:
    running = Server(flags=("--max-segment-seconds", "1.0"))
    try:
        pcm = pcm16("0006_en_short")
        with connect(f"ws://127.0.0.1:{running.port}/v1/realtime") as socket:
            update_session(socket)
            append_pcm(socket, pcm)
            socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            socket.send(json.dumps({"type": "transcription.done"}))
            events = receive_until(socket, "transcription.completed")
    finally:
        running.stop()
    finals = final_segments(events)
    # Note (VLa1111): the 3.05 s clip covers three full segments and a remainder.
    assert [segment["segment_id"] for segment in finals] == [0, 1, 2, 3]
    assert events[-1]["text"] == joined_text(finals)


def test_pure_silence_finalizes_with_empty_text(server: Server) -> None:
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        update_session(socket)
        append_pcm(socket, pcm16("0376_silence"))
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(json.dumps({"type": "transcription.done"}))
        events = receive_until(socket, "transcription.completed")
    assert [segment["text"] for segment in final_segments(events)] == [""]
    assert events[-1]["text"] == ""


def test_leading_and_trailing_silence_keep_the_transcript(server: Server) -> None:
    silence = bytes(32000 * 2)
    pcm = silence + pcm16("0006_en_short") + silence
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        update_session(socket)
        append_pcm(socket, pcm)
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(json.dumps({"type": "transcription.done"}))
        events = receive_until(socket, "transcription.completed")
    assert len(final_segments(events)) == 1
    assert "thinking" in events[-1]["text"]


def test_short_speech_is_finalized(server: Server) -> None:
    # Note (VLa1111): 100 ms from the middle of the clip, so the peak is speech.
    pcm = pcm16("0006_en_short")[32000:35200]
    with connect(f"ws://127.0.0.1:{server.port}/v1/realtime") as socket:
        update_session(socket)
        append_pcm(socket, pcm)
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        socket.send(json.dumps({"type": "transcription.done"}))
        events = receive_until(socket, "transcription.completed")
    finals = final_segments(events)
    assert len(finals) == 1
    assert events[-1]["text"] == finals[0]["text"]


def test_end_of_stdin_stops_the_server() -> None:
    running = Server()
    running.process.stdin.close()
    assert running.process.wait(timeout=10) == 0


def test_shutdown_reports_stopped() -> None:
    running = Server()
    running.process.stdin.write('{"command": "shutdown"}\n')
    running.process.stdin.flush()
    assert json.loads(running.process.stdout.readline()) == {"event": "stopped"}
    assert running.process.wait(timeout=10) == 0


def test_unknown_model_kinds_are_rejected() -> None:
    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "qwen3_asr_server"),
            "--supervised",
            "--model-kind",
            "whisper",
            "--model-directory",
            "x",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max-segment-seconds", "0"),
        ("--max-segment-seconds", "0.00001"),
        ("--decode-interval-ms", "0"),
        ("--first-decode-ms", "-1"),
    ],
)
def test_invalid_realtime_settings_fail_at_startup(flag: str, value: str) -> None:
    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "qwen3_asr_server"),
            "--model-directory",
            "x",
            flag,
            value,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 2
    assert flag in completed.stderr
