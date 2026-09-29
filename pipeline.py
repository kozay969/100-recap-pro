"""Rednote -> Burmese dubbing pipeline (pure functions, no Streamlit dependency).

Stages:
  1. download_video   - yt-dlp (Rednote / xhslink / direct mp4)
  2. transcribe       - faster-whisper (Chinese)
  3. translate_llm    - OpenAI-compatible LLM -> spoken Burmese
  4. balance_groups   - DP time-balanced grouping of segments
  5. synthesize_group - Edge TTS (my-MM-ThihaNeural / NilarNeural)
  6. assemble         - atempo fit + ducked original + mux -> final MP4
"""
import asyncio
import json
import math
import os
import re
import subprocess
import tempfile
import time

import httpx
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

FFMPEG = get_ffmpeg_exe()
SR = 44100

# Calibrated on a real 7:15 video: Thiha @ rate 1.0 ~= 11.7 Burmese chars/sec.
BURMESE_CHARS_PER_SEC = 11.5
MAX_TEMPO = 1.6  # never speed narration beyond this; warn instead


# ---------------------------------------------------------------- utilities
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FAILED: {' '.join(cmd)}\n{r.stderr[-2000:]}")
    return r


def media_duration(path):
    # NOTE: `ffmpeg -i` (no output) exits 1 but still prints stream info.
    r = subprocess.run([FFMPEG, "-i", path], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
    if not m:
        raise RuntimeError(f"could not probe duration of {path}")
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


# ---------------------------------------------------------------- 1. download
def download_video(url, out_dir, cookies_path=None, progress_cb=None):
    """Download video with yt-dlp. Returns {'path','title','duration'}."""
    import yt_dlp

    os.makedirs(out_dir, exist_ok=True)
    tmpl = os.path.join(out_dir, "source.%(ext)s")

    def hook(d):
        if progress_cb and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            if total:
                progress_cb(d.get("downloaded_bytes", 0) / total)

    opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": tmpl,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "noplaylist": True,
    }
    if cookies_path:
        opts["cookiefile"] = cookies_path
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    path = os.path.join(out_dir, "source.mp4")
    if not os.path.exists(path):  # merged under a different name?
        cands = [f for f in os.listdir(out_dir) if f.startswith("source.")]
        if not cands:
            raise RuntimeError("download finished but no file found")
        path = os.path.join(out_dir, cands[0])
    return {
        "path": path,
        "title": info.get("title", "video"),
        "duration": info.get("duration") or media_duration(path),
    }


# ---------------------------------------------------------------- 2. transcribe
def transcribe(video_path, model_size="small", progress_cb=None):
    """Chinese transcription with faster-whisper. Returns [{start,end,text}]."""
    from faster_whisper import WhisperModel

    wav = os.path.join(tempfile.gettempdir(), "dubstudio_audio.wav")
    run([FFMPEG, "-y", "-v", "error", "-i", video_path,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav])
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        wav, language="zh", vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500))
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2),
             "text": s.text.strip()} for s in segments]
    if progress_cb:
        progress_cb(1.0)
    try:
        os.remove(wav)
    except OSError:
        pass
    return segs


# ---------------------------------------------------------------- 3. translate
TRANSLATE_SYSTEM = (
    "You are a professional Chinese-to-Burmese translator for movie recap videos. "
    "Translate each line into natural SPOKEN Burmese (စကားပြောဘာသာ), the way a "
    "Burmese narrator would speak it - warm, lively, easy to listen to. "
    "Keep names/terms consistent. Do NOT add explanations. "
    "Output ONLY the translated lines, one per line, in the same order, "
    "with the same numbering as the input."
)


def _describe_http_error(e):
    """Human-readable error including the API's response body."""
    try:
        body = e.response.text[:400]
    except Exception:  # noqa: BLE001
        body = ""
    return RuntimeError(f"API error {e.response.status_code}: {body}")


def _chat(base_url, api_key, model, messages, timeout=120, max_retries=6):
    """OpenAI-compatible chat call with exponential-backoff retry.

    429 (rate limit) and 5xx errors are retried automatically, honoring
    the server's Retry-After header when present.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    delay = 5.0
    for attempt in range(max_retries + 1):
        try:
            r = httpx.post(url, headers={"Authorization": f"Bearer {api_key}"},
                           json={"model": model, "messages": messages,
                                 "temperature": 0.3}, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            retryable = status == 429 or 500 <= status < 600
            if not retryable or attempt == max_retries:
                raise _describe_http_error(e) from e
            wait = delay
            ra = e.response.headers.get("retry-after")
            if ra:
                try:
                    wait = max(wait, float(ra))
                except ValueError:
                    pass
            time.sleep(wait)
            delay = min(delay * 2, 90)
    raise RuntimeError("unreachable")  # pragma: no cover


def test_llm_connection(base_url, api_key, model, timeout=30):
    """Tiny probe call. Returns (ok: bool, message: str)."""
    try:
        out = _chat(base_url, api_key, model,
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    timeout=timeout, max_retries=0)
        return True, f"ချိတ်ဆက်အောင်မြင်ပါတယ် ✅ (model: {out.strip()[:80]})"
    except Exception as e:  # noqa: BLE001 - surface any failure as message
        return False, f"မအောင်မြင်ပါ: {e}"


def translate_llm(segments, api_key, base_url, model,
                  batch_size=20, progress_cb=None, delay_between=2.0):
    """Translate segment texts to spoken Burmese. Returns [burmese_text].

    Batches requests and paces them (delay_between seconds) to stay under
    API rate limits; 429/5xx responses are retried with backoff.
    """
    out = []
    total = len(segments)
    for i in range(0, total, batch_size):
        chunk = segments[i:i + batch_size]
        numbered = "\n".join(f"{j+1}. {s['text']}"
                             for j, s in enumerate(chunk))
        content = _chat(
            base_url, api_key, model,
            [{"role": "system", "content": TRANSLATE_SYSTEM},
             {"role": "user",
              "content": f"Translate these {len(chunk)} lines:\n{numbered}"}])
        lines = []
        for raw in content.strip().split("\n"):
            raw = raw.strip()
            if not raw:
                continue
            m = re.match(r"^\d+[.)\s\-:]+(.*)$", raw)
            lines.append(m.group(1).strip() if m else raw)
        # tolerate minor count mismatch: pad/truncate, never silently shift
        while len(lines) < len(chunk):
            lines.append(chunk[len(lines)]["text"])  # fallback: original
        out.extend(lines[:len(chunk)])
        if progress_cb:
            progress_cb(len(out) / total)
        if i + batch_size < total and delay_between > 0:
            time.sleep(delay_between)
    return out


# ---------------------------------------------------------------- 4. balance groups
def estimate_narration_secs(burmese_texts):
    chars = sum(len(t) for t in burmese_texts)
    return chars / BURMESE_CHARS_PER_SEC


def balance_groups(segments, burmese_texts, n_groups=4):
    """Partition segments into n_groups minimizing the worst narration rate.

    DP with lexicographic cost (max_group_rate, balance_penalty): the primary
    goal is that no group needs extreme speed-up; the tie-break keeps groups
    balanced. Returns [(start_idx, end_idx), ...].
    """
    n = len(segments)
    if n == 0:
        raise ValueError("no segments to group")
    n_groups = max(1, min(n_groups, n))
    est = [len(t) / BURMESE_CHARS_PER_SEC for t in burmese_texts]
    min_size = max(2, n // (n_groups * 3))
    pe, ps = [0.0], [0.0]
    for e, s in zip(est, segments):
        pe.append(pe[-1] + e)
        ps.append(ps[-1] + max(s["end"] - s["start"], 0.01))

    def rate(j, i):  # segments j..i inclusive
        e = pe[i + 1] - pe[j]
        sp = ps[i + 1] - ps[j]
        return e / sp

    INF = float("inf")
    # dp[k][i] = best (max_rate, penalty) for first i segments in k groups
    dp = [[(INF, INF)] * (n + 1) for _ in range(n_groups + 1)]
    par = [[0] * (n + 1) for _ in range(n_groups + 1)]
    dp[0][0] = (0.0, 0.0)
    for k in range(1, n_groups + 1):
        for i in range(k * min_size, n + 1):
            best, bj = (INF, INF), 0
            for j in range((k - 1) * min_size, i - min_size + 1):
                prev = dp[k - 1][j]
                if prev[0] == INF:
                    continue
                r = rate(j, i - 1)
                cand = (max(prev[0], r),
                        prev[1] + (r - 1.0) ** 2 * (ps[i] - ps[j]))
                if cand < best:
                    best, bj = cand, j
            dp[k][i] = best
            par[k][i] = bj
    groups, i = [], n
    for k in range(n_groups, 0, -1):
        j = par[k][i]
        groups.append((j, i - 1))
        i = j
    groups.reverse()
    return groups


# ---------------------------------------------------------------- 5. TTS
def _split_text(text, limit=1800):
    """Split long text on sentence boundaries for reliable synthesis."""
    parts, cur = [], ""
    for piece in re.split(r"([။.!?？！\n])", text):
        if len(cur) + len(piece) > limit and cur.strip():
            parts.append(cur.strip())
            cur = ""
        cur += piece
    if cur.strip():
        parts.append(cur.strip())
    # merge orphan punctuation / tiny fragments into the previous part
    merged = []
    for p in parts:
        if merged and len(p) < 30:
            merged[-1] = (merged[-1] + " " + p).strip()
        else:
            merged.append(p)
    return merged or [text]


def synthesize_group(text, voice, rate_pct, out_path, retries=2):
    """Edge TTS one narration group -> mp3. rate_pct e.g. 0, +10, -10."""
    import edge_tts

    async def _run():
        rate = f"{rate_pct:+d}%" if rate_pct else "+0%"
        parts = _split_text(text)
        tmpdir = tempfile.mkdtemp(prefix="dubtts_")
        mp3s = []
        for idx, part in enumerate(parts):
            p = os.path.join(tmpdir, f"p{idx}.mp3")
            tts = edge_tts.Communicate(part, voice, rate=rate)
            await tts.save(p)
            mp3s.append(p)
        if len(mp3s) == 1:
            os.replace(mp3s[0], out_path)
        else:
            lst = os.path.join(tmpdir, "list.txt")
            with open(lst, "w") as f:
                for p in mp3s:
                    f.write(f"file '{p}'\n")
            run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                 "-i", lst, "-c", "copy", out_path])

    last = None
    for _ in range(retries + 1):
        try:
            asyncio.run(_run())
            return out_path
        except Exception as e:  # noqa: BLE001 - retry transient TTS errors
            last = e
    raise RuntimeError(f"TTS failed after {retries + 1} tries: {last}")


# ---------------------------------------------------------------- 6. assemble
def _read_wav_mono(path):
    import wave
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 \
            and w.getframerate() == SR, path
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _write_wav_mono(path, audio):
    import wave
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def assemble(video_path, segments, groups, group_mp3s,
             out_path, duck=0.12, progress_cb=None):
    """Fit each group narration to its span, mix with ducked original, mux.

    Returns {'path','duration','rates','warnings'}.
    """
    tmp = tempfile.mkdtemp(prefix="dubasm_")
    video_dur = media_duration(video_path)
    total_n = int(video_dur * SR) + SR
    dub = np.zeros(total_n, dtype=np.float32)
    rates, warnings = [], []

    for gi, ((a, b), mp3) in enumerate(zip(groups, group_mp3s)):
        wav = os.path.join(tmp, f"g{gi}.wav")
        run([FFMPEG, "-y", "-v", "error", "-i", mp3,
             "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", wav])
        narr = _read_wav_mono(wav)
        narr_dur = len(narr) / SR
        span = segments[b]["end"] - segments[a]["start"]
        rate = narr_dur / span if span > 0 else 1.0
        if rate > MAX_TEMPO:
            warnings.append(
                f"Group {gi+1}: narration {narr_dur:.0f}s vs span {span:.0f}s "
                f"needs {rate:.2f}x - capped at {MAX_TEMPO}x, may overflow")
            rate = MAX_TEMPO
        rates.append(round(rate, 3))
        if abs(rate - 1.0) > 0.005:
            fit_wav = os.path.join(tmp, f"g{gi}_fit.wav")
            run([FFMPEG, "-y", "-v", "error", "-i", wav,
                 "-filter:a", f"atempo={rate:.4f}",
                 "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", fit_wav])
            fit = _read_wav_mono(fit_wav)
        else:
            fit = narr
        target_n = int(span * SR)
        if len(fit) > target_n:
            fit = fit[:target_n]
        pos = int((segments[a]["start"] + 0.02) * SR)
        end = min(pos + len(fit), total_n)
        dub[pos:end] += fit[:end - pos]
        if progress_cb:
            progress_cb((gi + 1) / len(groups) * 0.7)

    orig_wav = os.path.join(tmp, "orig.wav")
    run([FFMPEG, "-y", "-v", "error", "-i", video_path,
         "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", orig_wav])
    orig = _read_wav_mono(orig_wav)
    n = min(len(orig), total_n)
    mix = dub.copy()
    mix[:n] += orig[:n] * duck
    peak = float(np.max(np.abs(mix)))
    if peak > 0.98:
        mix *= 0.98 / peak
    mix_wav = os.path.join(tmp, "mix.wav")
    _write_wav_mono(mix_wav, mix[:int(video_dur * SR)])
    if progress_cb:
        progress_cb(0.85)

    run([FFMPEG, "-y", "-v", "error", "-i", video_path, "-i", mix_wav,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-map", "0:v:0", "-map", "1:a:0", "-shortest", out_path])
    if progress_cb:
        progress_cb(1.0)
    return {"path": out_path, "duration": media_duration(out_path),
            "rates": rates, "warnings": warnings,
            "mix_peak": round(peak, 3)}


def save_project_json(path, segments, burmese_texts):
    data = {"segments": [
        {"start": s["start"], "end": s["end"],
         "original_text": s["text"], "burmese_text": b}
        for s, b in zip(segments, burmese_texts)]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def load_project_json(path):
    """Flexible loader: accepts
      {"segments": [{start,end,original_text,burmese_text}...]},
      {"translations": [...]} (same objects, or plain burmese strings),
      or a raw list of either.
    Returns (segments|None, burmese_list)."""
    data = json.load(open(path, encoding="utf-8"))
    items = data.get("segments") or data.get("translations") \
        if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("unrecognized project JSON format")
    if items and isinstance(items[0], str):
        return None, [str(x) for x in items]  # burmese lines only
    segs = [{"start": it["start"], "end": it["end"],
             "text": it.get("original_text", it.get("text", ""))}
            for it in items]
    burmese = [it.get("burmese_text", "") for it in items]
    return segs, burmese
