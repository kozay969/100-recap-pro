"""Rednote -> Burmese dubbing pipeline (pure functions, no Streamlit dependency).

Stages:
  1. download_video   - yt-dlp (Rednote / xhslink / direct mp4)
  2. transcribe       - faster-whisper (Chinese)
  3. translate_llm    - OpenAI-compatible LLM -> spoken Burmese
  4. balance_groups   - DP time-balanced grouping of segments
  5. synthesize_lines - Edge TTS per line (my-MM-ThihaNeural / NilarNeural)
  6. assemble         - per-line placement at own timestamps + ducked
                        original + mux -> final MP4
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
MAX_TEMPO = 2.0  # atempo's limit; faster than this sounds unnatural


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

# Ko Zay's preferred prompt: strict JSON output, inputs labelled
# as  ID: <n> | Duration: <d>s | Text: "<chinese>"
TRANSLATE_SYSTEM_JSON = (
    "You are a professional video dubbing translator. You MUST translate "
    "the following Chinese subtitles into natural spoken Burmese "
    "(Myanmar script ONLY, NO English, NO phonetic guides).\n\n"
    "Translate each line so that it sounds natural in a movie recap narration. "
    "Do not translate word-for-word.\n\n"
    "Return the result STRICTLY as a JSON object with a single key "
    "\"translations\" containing an array of strings in order.\n"
    "Example:\n"
    "{\n"
    '  "translations": [\n'
    "    \"မြန်မာစာသား ၁\",\n"
    "    \"မြန်မာစာသား ၂\"\n"
    "  ]\n"
    "}"
)


def _describe_http_error(e):
    """Human-readable error including the API's response body."""
    try:
        body = e.response.text[:400]
    except Exception:  # noqa: BLE001
        body = ""
    return RuntimeError(f"API error {e.response.status_code}: {body}")


def _with_retry(fn, max_retries=8):
    """Run fn(); retry 429/5xx (honoring Retry-After) and transient
    network errors with exponential backoff."""
    delay = 5.0
    attempt = 0
    while True:
        try:
            return fn()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            retryable = status == 429 or 500 <= status < 600
            if not retryable or attempt >= max_retries:
                raise _describe_http_error(e) from e
            wait = delay
            ra = e.response.headers.get("retry-after")
            if ra:
                try:
                    wait = max(wait, float(ra))
                except ValueError:
                    pass
        except (httpx.TimeoutException, httpx.ConnectError):
            if attempt >= max_retries:
                raise
            wait = delay
        time.sleep(wait)
        delay = min(delay * 2, 120)
        attempt += 1


def _openai_once(base_url, api_key, model, messages, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    r = httpx.post(url, headers={"Authorization": f"Bearer {api_key}"},
                   json={"model": model, "messages": messages,
                         "temperature": 0.3}, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _gemini_native_once(api_key, model, messages, timeout):
    """Native Gemini REST API (generateContent) — the officially
    supported path for free AI Studio keys."""
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    system = "\n".join(m["content"] for m in messages
                       if m["role"] == "system")
    contents = [{"role": "user", "parts": [{"text": m["content"]}]} for m in messages
                if m["role"] != "system"]
    body = {"contents": contents, "generationConfig": {"temperature": 0.3}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = httpx.post(url, headers={"x-goog-api-key": api_key}, json=body,
                   timeout=timeout)
    r.raise_for_status()
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(
            f"unexpected Gemini response: {str(data)[:400]}")


def _chat(base_url, api_key, model, messages, timeout=120, max_retries=8):
    """Chat call with exponential-backoff retry.

    base_url == "gemini-native" uses the native Gemini REST API;
    anything else uses an OpenAI-compatible /chat/completions endpoint.
    """
    if base_url == "gemini-native":
        fn = lambda: _gemini_native_once(api_key, model, messages, timeout)  # noqa: E731
    else:
        fn = lambda: _openai_once(base_url, api_key, model, messages, timeout)  # noqa: E731
    return _with_retry(fn, max_retries)


def test_llm_connection(base_url, api_key, model, timeout=30):
    """Tiny probe call. Returns (ok: bool, message: str)."""
    try:
        out = _chat(base_url, api_key, model,
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    timeout=timeout, max_retries=0)
        return True, f"ချိတ်ဆက်အောင်မြင်ပါတယ် ✅ (model: {out.strip()[:80]})"
    except Exception as e:  # noqa: BLE001 - surface any failure as message
        return False, f"မအောင်မြင်ပါ: {e}"


def _parse_translation_output(content, n):
    """Parse LLM translation output -> exactly n strings.

    Accepts {"translations": [...]} JSON (bare or inside ```json fences),
    or one-per-line text with optional numbering. Short output is padded
    with "" (silence) rather than the untranslated original, so a missing
    line never ends up spoken in the wrong language.
    """
    txt = content.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", txt)
    if m:
        txt = m.group(1).strip()
    if txt.startswith("{"):
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict) and isinstance(
                    obj.get("translations"), list):
                lines = [str(x).strip() for x in obj["translations"]]
                while len(lines) < n:
                    lines.append("")
                return lines[:n]
        except Exception:  # noqa: BLE001
            pass
    lines = []
    for raw in content.strip().split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        m2 = re.match(r"^\d+[.)\s\-:]+(.*)$", raw)
        lines.append(m2.group(1).strip() if m2 else raw)
    # tolerate minor count mismatch: pad/truncate, never silently shift
    while len(lines) < n:
        lines.append("")
    return lines[:n]


def translate_llm(segments, api_key, base_url, model,
                  batch_size=20, progress_cb=None, delay_between=3.0,
                  done=None, checkpoint_cb=None, system=None,
                  input_style="numbered"):
    """Translate segment texts to spoken Burmese. Returns [burmese_text].

    Batches requests and paces them (delay_between seconds) to stay under
    API rate limits; 429/5xx responses and transient network errors are
    retried with backoff. `done` = already-translated lines to resume from;
    `checkpoint_cb(out)` is called after every batch so the caller can
    persist partial progress. `system` overrides the default system prompt.
    `input_style="id_duration"` sends lines as
    ID: <n> | Duration: <d>s | Text: "<chinese>" (Ko Zay's JSON-prompt format).
    """
    sys_prompt = system or TRANSLATE_SYSTEM
    out = list(done) if done else []
    total = len(segments)
    for i in range(len(out), total, batch_size):
        chunk = segments[i:i + batch_size]
        if input_style == "id_duration":
            lines_in = "\n".join(
                f"ID: {i + j} | Duration: "
                f"{s.get('end', 0) - s.get('start', 0):.1f}s | "
                f"Text: \"{s['text']}\""
                for j, s in enumerate(chunk))
            user_content = f"Inputs:\n{lines_in}"
        else:
            lines_in = "\n".join(f"{j + 1}. {s['text']}"
                                 for j, s in enumerate(chunk))
            user_content = (f"Translate these {len(chunk)} lines:\n"
                            f"{lines_in}")
        content = _chat(
            base_url, api_key, model,
            [{"role": "system", "content": sys_prompt},
             {"role": "user", "content": user_content}])
        lines = _parse_translation_output(content, len(chunk))
        out.extend(lines)
        if checkpoint_cb:
            checkpoint_cb(list(out))
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


def synthesize_line(text, voice, rate_pct, out_path, retries=2):
    """Edge TTS one subtitle line -> mp3. rate_pct e.g. 0, +10, -10."""
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


def synthesize_lines(texts, voice, rate_pct, out_dir, progress_cb=None,
                     workers=4):
    """Synthesize every line (in parallel) -> [mp3 paths].

    Per-line synthesis is what keeps the dubbing in sync: each line is
    later placed at its own timestamp instead of stretching a whole
    group uniformly.
    """
    import concurrent.futures
    os.makedirs(out_dir, exist_ok=True)
    paths = [os.path.join(out_dir, f"line_{i:04d}.mp3")
             for i in range(len(texts))]

    def _one(i):
        synthesize_line(texts[i], voice, rate_pct, paths[i])
        return i

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers) as ex:
        futs = {ex.submit(_one, i): i for i in range(len(texts))}
        done = 0
        for f in concurrent.futures.as_completed(futs):
            f.result()  # re-raise the first failure, if any
            done += 1
            if progress_cb:
                progress_cb(done / len(texts))
    return paths


def synthesize_selected(idxs, texts, voice, rate_pct, paths,
                        progress_cb=None, workers=4):
    """Re-synthesize a subset of lines (by index) at a different TTS speed,
    e.g. to fix lines whose narration overflows their time slot."""
    import concurrent.futures

    def _one(k, i):
        synthesize_line(texts[i], voice, rate_pct, paths[i])
        return k

    idxs = list(idxs)
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers) as ex:
        futs = {ex.submit(_one, k, i): i for k, i in enumerate(idxs)}
        for k, f in enumerate(concurrent.futures.as_completed(futs)):
            f.result()  # re-raise the first failure, if any
            if progress_cb:
                progress_cb((k + 1) / len(idxs))


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


def _has_audio(path):
    r = subprocess.run([FFMPEG, "-i", path],
                       capture_output=True, text=True)
    return "Audio:" in r.stderr


def _video_fps(path):
    """Source frame rate (for fps-normalizing retimed pieces)."""
    r = subprocess.run([FFMPEG, "-i", path],
                       capture_output=True, text=True)
    m = re.search(r"(\d+(?:\.\d+)?)\s+fps", r.stderr)
    try:
        return float(m.group(1)) if m else 30.0
    except Exception:
        return 30.0


def _fade_edges(audio, ms=8):
    """Tiny fade in/out so back-to-back clips never click."""
    n = int(SR * ms / 1000)
    if len(audio) < 2 * n or n <= 0:
        return audio
    fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = audio.copy()
    out[:n] *= fade
    out[-n:] *= fade[::-1]
    return out


def _place_narration(tmp, placements, total_dur,
                     progress_cb=None, p0=0.0, p1=1.0):
    """Lay narration clips on a timeline.

    placements: list of (t_start, wav_path, tempo, max_dur).
    Each clip starts EXACTLY at t_start (no offset -> no overlap with the
    next clip), is atempo'd only when tempo > 1.005, gets 8ms edge fades,
    and is truncated to max_dur (its piece's real duration) so nothing
    bleeds into the next piece. Returns (dub_array, [(start, end), ...]).
    """
    total_n = int(total_dur * SR) + SR
    dub = np.zeros(total_n, dtype=np.float32)
    placed = []
    m = max(len(placements), 1)
    for j, (t_start, wav, tempo, max_dur) in enumerate(placements):
        if tempo > 1.005:
            fit_wav = os.path.join(tmp, f"plc{j}_fit.wav")
            run([FFMPEG, "-y", "-v", "error", "-i", wav,
                 "-filter:a", f"atempo={tempo:.4f}",
                 "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", fit_wav])
            clip = _read_wav_mono(fit_wav)
        else:
            clip = _read_wav_mono(wav)
        clip = clip[:max(1, int(max_dur * SR))]
        clip = _fade_edges(clip)
        pos = int(t_start * SR)
        end = min(pos + len(clip), total_n)
        if end > pos:
            dub[pos:end] += clip[:end - pos]
            placed.append((t_start, t_start + (end - pos) / SR))
        else:
            placed.append((t_start, t_start))
        if progress_cb:
            progress_cb(p0 + (p1 - p0) * (j + 1) / m)
    return dub, placed


def assemble(video_path, segments, line_mp3s,
             out_path, duck=0.12, progress_cb=None):
    """Place each line's narration at its own timestamp, mix, mux.

    Per-line placement (instead of stretching whole groups) is what keeps
    the dubbing in sync with the video: a line always starts on time, and
    any overflow only eats into the pause after it. Lines shorter than
    their slot simply end early (natural pause) - never slowed down.

    Returns {'path','duration','rates','warnings'}.
    """
    tmp = tempfile.mkdtemp(prefix="dubasm_")
    video_dur = media_duration(video_path)
    total_n = int(video_dur * SR) + SR
    dub = np.zeros(total_n, dtype=np.float32)
    rates, warnings = [], []
    n_lines = len(segments)

    for i, (seg, mp3) in enumerate(zip(segments, line_mp3s)):
        wav = os.path.join(tmp, f"l{i}.wav")
        run([FFMPEG, "-y", "-v", "error", "-i", mp3,
             "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", wav])
        narr = _read_wav_mono(wav)
        narr_dur = len(narr) / SR
        slot = max(seg["end"] - seg["start"], 0.01)
        next_start = segments[i + 1]["start"] if i + 1 < n_lines else video_dur
        # available window: own slot + the pause after it (borrowed only if
        # needed - the next line still starts exactly on time)
        avail = max(next_start - seg["start"], slot)
        rate_slot = narr_dur / slot
        if rate_slot <= MAX_TEMPO:
            target, tempo = slot, rate_slot
        elif narr_dur / avail <= MAX_TEMPO:
            target, tempo = avail, narr_dur / avail
        else:
            target, tempo = avail, MAX_TEMPO
            warnings.append(
                f"Line {i+1}: {narr_dur:.1f}s audio needs {rate_slot:.2f}x "
                f"but capped at {MAX_TEMPO}x - cut at next line; "
                f"consider shorter text")
        rates.append(round(tempo, 3))
        if tempo > 1.005:
            fit_wav = os.path.join(tmp, f"l{i}_fit.wav")
            run([FFMPEG, "-y", "-v", "error", "-i", wav,
                 "-filter:a", f"atempo={tempo:.4f}",
                 "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", fit_wav])
            fit = _read_wav_mono(fit_wav)
        else:
            fit = narr  # short line: ends early, natural pause follows
        target_n = int(target * SR)
        if len(fit) > target_n:
            fit = fit[:target_n]
        # absolute no-overlap guarantee: never write past next line's start
        stop_n = int(min(next_start, video_dur) * SR) \
            if i + 1 < n_lines else total_n
        pos = int((seg["start"] + 0.02) * SR)
        end = min(pos + len(fit), stop_n, total_n)
        if end > pos:
            dub[pos:end] += fit[:end - pos]
        if progress_cb:
            progress_cb((i + 1) / n_lines * 0.7)

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


def assemble_freeze(video_path, segments, line_mp3s, out_path,
                    max_tempo=1.3, progress_cb=None):
    """Fit the VIDEO to the narration (freeze-frame extension).

    Narration plays at natural speed (sped up at most max_tempo);
    wherever it still overflows its slot, the video's last frame is
    frozen to make room. No truncation, no overlap, no chipmunk audio.
    Trade-off: the video gets longer, and the original audio is dropped
    (a freeze would break its continuity).

    Returns {"out","new_duration","added_sec","extensions","rates",
             "warnings","new_segments"} - new_segments carries the
    shifted timestamps (use these for SRT in freeze mode).
    """
    tmp = tempfile.mkdtemp(prefix="dubfreeze_")
    n = len(segments)
    if len(line_mp3s) != n:
        raise ValueError(f"line_mp3s ({len(line_mp3s)}) != segments ({n})")

    # 1. narration durations + per-line tempo / extension plan
    durs, tempos, exts = [], [], []
    for i, mp3 in enumerate(line_mp3s):
        wav = os.path.join(tmp, f"l{i}.wav")
        run([FFMPEG, "-y", "-v", "error", "-i", mp3,
             "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", wav])
        d = media_duration(wav)
        durs.append(d)
        slot = max(segments[i]["end"] - segments[i]["start"], 0.01)
        need = d / slot
        if need <= 1.0:
            tempos.append(1.0)
            exts.append(0.0)
        elif need <= max_tempo:
            tempos.append(need)
            exts.append(0.0)
        else:
            tempos.append(max_tempo)
            exts.append(d / max_tempo - slot)

    # 2. retimed video: segment pieces (frozen at the end if needed),
    #    gaps and head/tail untouched
    video_dur = media_duration(video_path)
    ops = []  # ("gap", dur) | ("seg", idx, dur, ext)
    prev_end = 0.0
    for i, seg in enumerate(segments):
        s, e = seg["start"], seg["end"]
        if s > prev_end + 0.001:
            ops.append(("gap", s - prev_end))
        ops.append(("seg", i, e - s, exts[i]))
        prev_end = e
    if prev_end < video_dur - 0.001:
        ops.append(("gap", video_dur - prev_end))

    piece_files = []
    piece_durs = []  # measured AFTER encode: re-encode frame rounding
    t_src = 0.0      # means planned sums drift from reality; the dub
    for k, op in enumerate(ops):  # track MUST follow measured durations
        if op[0] == "gap":
            vs, vd, ext = t_src, op[1], 0.0
        else:
            _, i, vd, ext = op
            vs = segments[i]["start"]
        p = os.path.join(tmp, f"p{k:04d}.mp4")
        cmd = [FFMPEG, "-y", "-v", "error",
               "-ss", f"{vs:.3f}", "-t", f"{vd:.3f}", "-i", video_path]
        if ext > 0.02:
            cmd += ["-filter:v",
                    f"tpad=stop_mode=clone:stop_duration={ext:.3f}"]
        cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-an", p]
        run(cmd)
        piece_files.append(p)
        piece_durs.append(media_duration(p))
        t_src = vs + vd
        if progress_cb:
            progress_cb((k + 1) / len(ops) * 0.4)

    lst = os.path.join(tmp, "pieces.txt")
    with open(lst, "w") as f:
        for p in piece_files:
            f.write(f"file '{p}'\n")
    vretimed = os.path.join(tmp, "video_retimed.mp4")
    run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", lst, "-c", "copy", vretimed])
    new_dur = media_duration(vretimed)
    added_sec = new_dur - video_dur

    # 3. dub track on the MEASURED new timeline (no drift, no truncation,
    #    no overlap: clips start exactly on piece boundaries, edge-faded)
    placements, seg_order = [], []
    rates, warnings = [], []
    t = 0.0
    for k, op in enumerate(ops):
        pd = piece_durs[k]
        if op[0] == "gap":
            t += pd
            continue
        _, i, vd, ext = op
        rates.append(round(tempos[i], 3))
        placements.append((t, os.path.join(tmp, f"l{i}.wav"),
                           tempos[i], pd))
        seg_order.append(i)
        if ext > 3.0:
            warnings.append(f"Line {i+1}: video frozen {ext:.1f}s - "
                            f"long still frame")
        t += pd
    dub, placed = _place_narration(tmp, placements, new_dur,
                                   progress_cb, 0.4, 0.8)
    new_segments = [{"start": s, "end": e,
                     "text": segments[i]["text"]}
                    for (s, e), i in zip(placed, seg_order)]

    dub_wav = os.path.join(tmp, "dub.wav")
    _write_wav_mono(dub_wav, dub[:int(new_dur * SR)])
    run([FFMPEG, "-y", "-v", "error", "-i", vretimed, "-i", dub_wav,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-map", "0:v:0", "-map", "1:a:0", "-shortest", out_path])
    if progress_cb:
        progress_cb(1.0)
    return {"out": out_path, "new_duration": new_dur,
            "added_sec": added_sec,
            "extensions": sum(1 for e in exts if e > 0.02),
            "rates": rates, "warnings": warnings,
            "new_segments": new_segments}


def assemble_slowmo(video_path, segments, line_mp3s, out_path,
                    max_slow=1.5, duck=0.12, progress_cb=None):
    """Fit the VIDEO to the narration with smooth slow-motion.

    The narration ALWAYS plays at natural speed (never sped up). Wherever
    it overflows its slot, that video segment is slowed down smoothly
    (up to max_slow); any remaining overflow is covered by a freeze frame.
    The original audio is slowed to match the video and ducked quietly
    under the narration.

    Returns {"out","new_duration","added_sec","slowmos","extensions",
             "rates","slow_factors","warnings","new_segments"} -
    new_segments carries the shifted timestamps (use these for SRT).
    """
    tmp = tempfile.mkdtemp(prefix="dubslow_")
    n = len(segments)
    if len(line_mp3s) != n:
        raise ValueError(f"line_mp3s ({len(line_mp3s)}) != segments ({n})")
    if not 1.0 <= max_slow <= 2.0:
        raise ValueError("max_slow must be between 1.0 and 2.0")
    has_aud = _has_audio(video_path)
    src_fps = _video_fps(video_path)

    # 1. narration durations + per-line slow/freeze plan (audio: natural)
    factors, exts = [], []
    for i, mp3 in enumerate(line_mp3s):
        wav = os.path.join(tmp, f"l{i}.wav")
        run([FFMPEG, "-y", "-v", "error", "-i", mp3,
             "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", wav])
        d = media_duration(wav)
        slot = max(segments[i]["end"] - segments[i]["start"], 0.01)
        need = d / slot
        if need <= 1.0:
            factors.append(1.0)
            exts.append(0.0)
        elif need <= max_slow:
            # smooth slow-mo exactly covers the narration
            factors.append(need)
            exts.append(0.0)
        else:
            # slow as much as allowed, freeze the rest
            factors.append(max_slow)
            exts.append(d - slot * max_slow)

    # 2. retimed pieces: slowed video (+slowed original audio), frozen
    #    tail where slow-mo alone is not enough; gaps untouched
    video_dur = media_duration(video_path)
    ops = []  # ("gap", dur) | ("seg", idx, dur, factor, ext)
    prev_end = 0.0
    for i, seg in enumerate(segments):
        s, e = seg["start"], seg["end"]
        if s > prev_end + 0.001:
            ops.append(("gap", s - prev_end))
        ops.append(("seg", i, e - s, factors[i], exts[i]))
        prev_end = e
    if prev_end < video_dur - 0.001:
        ops.append(("gap", video_dur - prev_end))

    piece_files, piece_durs = [], []
    t_src = 0.0
    for k, op in enumerate(ops):
        if op[0] == "gap":
            vs, vd, fac, ext = t_src, op[1], 1.0, 0.0
        else:
            _, i, vd, fac, ext = op
            vs = segments[i]["start"]
        p = os.path.join(tmp, f"p{k:04d}.mp4")
        cmd = [FFMPEG, "-y", "-v", "error",
               "-ss", f"{vs:.3f}", "-t", f"{vd:.3f}", "-i", video_path]
        # Filter notes (verified against the bundled ffmpeg 7.0.2):
        #  * tpad AFTER setpts in one chain silently drops the padded
        #    frames -> freeze FIRST, pre-scaled (ext/fac), then slow down:
        #    (vd + ext/fac) * fac = vd*fac + ext.
        #  * tpad by FRAME COUNT (stop=N) is frame-exact; stop_duration
        #    can round to a slightly short piece.
        #  * setpts leaves sparse timestamps (e.g. 20fps-effective in a
        #    30fps stream) which corrupt -c copy concat when mixed with
        #    plain pieces -> fps=src_fps normalizes every slowed piece.
        vf = []
        if ext > 0.02:
            stop_frames = max(1, int(round(ext / fac * src_fps)))
            vf.append(f"tpad=stop_mode=clone:stop={stop_frames}")
        if fac > 1.001:
            vf.append(f"setpts={fac:.4f}*PTS")
            vf.append(f"fps={src_fps:g}")
        if vf:
            cmd += ["-filter:v", ",".join(vf)]
        if has_aud:
            af = []
            if fac > 1.001:
                # atempo keeps pitch, only stretches time (0.5..2.0 OK
                # because 1/max_slow >= 0.5)
                af.append(f"atempo={1.0 / fac:.4f}")
            if ext > 0.02:
                # pad original audio with silence so it matches the
                # (partly frozen) video piece length
                af.append(f"apad=whole_dur={vd * fac + ext:.3f}")
            if af:
                cmd += ["-filter:a", ",".join(af)]
            cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-c:a", "aac", "-b:a", "96k", p]
        else:
            cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                    "-an", p]
        run(cmd)
        piece_files.append(p)
        piece_durs.append(media_duration(p))
        t_src = vs + vd
        if progress_cb:
            progress_cb((k + 1) / len(ops) * 0.35)

    lst = os.path.join(tmp, "pieces.txt")
    with open(lst, "w") as f:
        for p in piece_files:
            f.write(f"file '{p}'\n")
    vretimed = os.path.join(tmp, "video_retimed.mp4")
    run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0",
         "-i", lst, "-c", "copy", vretimed])
    new_dur = media_duration(vretimed)
    added_sec = new_dur - video_dur

    # 3. narration on the measured timeline, always at natural speed
    placements, seg_order = [], []
    rates, slow_factors, warnings = [], [], []
    t = 0.0
    for k, op in enumerate(ops):
        pd = piece_durs[k]
        if op[0] == "gap":
            t += pd
            continue
        _, i, vd, fac, ext = op
        rates.append(1.0)
        slow_factors.append(round(fac, 3))
        placements.append((t, os.path.join(tmp, f"l{i}.wav"), 1.0, pd))
        seg_order.append(i)
        if ext > 3.0:
            warnings.append(f"Line {i+1}: slow-mo capped at {max_slow:.1f}x, "
                            f"video frozen {ext:.1f}s for the rest")
        t += pd
    dub, placed = _place_narration(tmp, placements, new_dur,
                                   progress_cb, 0.35, 0.7)
    new_segments = [{"start": s, "end": e,
                     "text": segments[i]["text"]}
                    for (s, e), i in zip(placed, seg_order)]

    # 4. duck the (retimed) original audio under the narration
    if has_aud:
        orig_wav = os.path.join(tmp, "orig.wav")
        run([FFMPEG, "-y", "-v", "error", "-i", vretimed,
             "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", orig_wav])
        orig = _read_wav_mono(orig_wav)
    else:
        orig = np.zeros(1, dtype=np.float32)
    mix = dub.copy()
    mm = min(len(orig), len(dub))
    mix[:mm] += orig[:mm] * duck
    peak = float(np.max(np.abs(mix)))
    if peak > 0.98:
        mix *= 0.98 / peak
    mix_wav = os.path.join(tmp, "mix.wav")
    _write_wav_mono(mix_wav, mix[:int(new_dur * SR)])
    if progress_cb:
        progress_cb(0.9)

    run([FFMPEG, "-y", "-v", "error", "-i", vretimed, "-i", mix_wav,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-map", "0:v:0", "-map", "1:a:0", "-shortest", out_path])
    if progress_cb:
        progress_cb(1.0)
    return {"out": out_path, "new_duration": new_dur,
            "added_sec": added_sec,
            "slowmos": sum(1 for f in factors if f > 1.001),
            "extensions": sum(1 for e in exts if e > 0.02),
            "rates": rates, "slow_factors": slow_factors,
            "warnings": warnings, "new_segments": new_segments}


def save_project_json(path, segments, burmese_texts):
    data = {"segments": [
        {"start": s["start"], "end": s["end"],
         "original_text": s["text"], "burmese_text": b}
        for s, b in zip(segments, burmese_texts)]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def _srt_ts(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(segments, burmese_texts):
    """Burmese subtitle file content (SRT) timed to the segments."""
    out = []
    for i, (seg, b) in enumerate(zip(segments, burmese_texts)):
        out.append(f"{i + 1}\n"
                   f"{_srt_ts(seg['start'])} --> {_srt_ts(seg['end'])}\n"
                   f"{b.strip()}\n")
    return "\n".join(out)


def write_srt(path, segments, burmese_texts):
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_srt(segments, burmese_texts))
    return path


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
