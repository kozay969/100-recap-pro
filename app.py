"""🎬 Rednote -> Burmese Dubbing Studio (Streamlit app).

Pipeline: video -> transcribe (Chinese) -> translate (Burmese, LLM)
       -> Edge TTS voiceover -> time-fit assemble -> final MP4.
Every stage is reviewable/editable; every stage can also be skipped
by importing your own file.
"""
import os
import tempfile

import streamlit as st

import pipeline as P

st.set_page_config(page_title="Dubbing Studio", page_icon="🎬", layout="wide")

# ------------------------------------------------------------ session state
def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

ss("workdir", tempfile.mkdtemp(prefix="dubstudio_"))
ss("video_path", None)
ss("video_title", "")
ss("video_dur", 0.0)
ss("segments", None)      # [{start,end,text}]
ss("burmese", None)       # [str]
ss("burmese_partial", [])  # [str] auto-saved progress of a running/failed translate
ss("groups", None)        # [(a,b)]
ss("line_mp3s", None)      # [path] per line
ss("line_durs", None)       # [float] measured TTS seconds per line
ss("final_path", None)
ss("final_info", None)

WD = st.session_state.workdir
VOICES = {"Thiha - ယောက်ျားလေး": "my-MM-ThihaNeural",
          "Nilar - မိန်းကလေး": "my-MM-NilarNeural"}

st.title("🎬 Dubbing Studio")
st.caption("Rednote ဗီဒီယို → မြန်မာ dubbing — link → စာသား → ဘာသာပြန် → အသံ → ဗီဒီယို")

step = st.sidebar.radio(
    "အဆင့်",
    ["1️⃣ ဗီဒီယို", "2️⃣ စာသားခွဲခြင်း", "3️⃣ ဘာသာပြန်ခြင်း",
     "4️⃣ အသံသွင်းခြင်း", "5️⃣ ပေါင်းစပ်ခြင်း"],
    index=0,
)
st.sidebar.divider()
st.sidebar.caption("💡 အဆင့်တိုင်းမှာ ကိုယ်ပိုင်ဖိုင်တင်ပြီး ကျော်သွားလို့ရပါတယ်")


def need_video():
    if not st.session_state.video_path:
        st.warning("အရင် 1️⃣ ဗီဒီယို အဆင့်မှာ ဗီဒီယိုထည့်ပါ")
        st.stop()


def _save_tmp(uploaded, path):
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path


# ============================================================ 1. VIDEO
if step.startswith("1"):
    st.header("1️⃣ ဗီဒီယို")
    tab_link, tab_upload = st.tabs(["🔗 Link နဲ့ဒေါင်းမယ်", "📤 ဖိုင်တင်မယ်"])

    with tab_link:
        url = st.text_input("Rednote / xhslink / ဗီဒီယို link",
                            placeholder="http://xhslink.com/o/...")
        cookies = st.file_uploader("cookies.txt (login လိုတဲ့ link ဆို)",
                                   type=["txt"],
                                   help="xiaohongshu.com မှာ login ဝင်ထားတဲ့ browser ကနေ "
                                        "'Get cookies.txt LOCALLY' extension နဲ့ထုတ်ထားတဲ့ဖိုင်")
        if st.button("⬇️ ဒေါင်းလုဒ်ဆွဲမယ်", type="primary", disabled=not url):
            cookies_path = None
            if cookies:
                cookies_path = os.path.join(WD, "cookies.txt")
                with open(cookies_path, "wb") as f:
                    f.write(cookies.getbuffer())
            bar = st.progress(0.0, "ဒေါင်းလုဒ်ဆွဲနေပါတယ်…")
            try:
                info = P.download_video(
                    url, os.path.join(WD, "dl"), cookies_path,
                    progress_cb=lambda x: bar.progress(min(x, 1.0)))
                st.session_state.video_path = info["path"]
                st.session_state.video_title = info["title"]
                st.session_state.video_dur = info["duration"]
                bar.progress(1.0)
                st.success(f"ရပြီ! {info['title'][:60]} "
                           f"({info['duration']:.0f} စက္ကန့်)")
                st.rerun()
            except Exception as e:
                st.error("ဒေါင်းလုဒ်မရပါ")
                st.code(str(e)[:800])
                st.info("💡 cookies.txt တင်ကြည့်ပါ၊ ဒါမှမဟုတ် ဗီဒီယိုဖိုင်ကို "
                        "တိုက်ရိုက်တင်လိုက်ပါ (📤 ဖိုင်တင်မယ် tab)")

    with tab_upload:
        up = st.file_uploader("MP4 ဗီဒီယိုဖိုင်", type=["mp4", "mov", "mkv"])
        if up and st.button("သုံးမယ်", type="primary"):
            p = os.path.join(WD, "source.mp4")
            with open(p, "wb") as f:
                f.write(up.getbuffer())
            st.session_state.video_path = p
            st.session_state.video_title = up.name
            st.session_state.video_dur = P.media_duration(p)
            st.success("ဗီဒီယိုရပြီ!")
            st.rerun()

    if st.session_state.video_path:
        st.divider()
        st.subheader("ဗီဒီယို")
        st.video(st.session_state.video_path)
        st.write(f"**{st.session_state.video_title}** — "
                 f"{st.session_state.video_dur:.0f} စက္ကန့် "
                 f"({st.session_state.video_dur/60:.1f} မိနစ်)")

# ============================================================ 2. TRANSCRIBE
elif step.startswith("2"):
    st.header("2️⃣ စာသားခွဲခြင်း (Chinese transcription)")
    need_video()

    if st.session_state.segments is None:
        c1, c2 = st.columns(2)
        with c1:
            model = st.selectbox("Whisper model",
                                 ["tiny", "base", "small", "medium"],
                                 index=2,
                                 help="ကြီးလေတိကျလေ၊ ကြာလေပေါ့။ "
                                      "အက်ပ်ပြိုရင် base/tiny သုံးကြည့်ပါ")
            if st.button("🎙️ Transcribe လုပ်မယ်", type="primary"):
                bar = st.progress(0.0, "transcribe လုပ်နေပါတယ်… (ပထမအကြိမ်ဆို model ဒေါင်းရလို့ကြာမယ်)")
                try:
                    segs = P.transcribe(
                        st.session_state.video_path, model,
                        progress_cb=lambda x: bar.progress(x))
                    st.session_state.segments = segs
                    st.session_state.burmese = None
                    st.session_state.burmese_partial = []
                    bar.progress(1.0)
                    st.success(f"စာကြောင်း {len(segs)} ကြောင်း ရပြီ!")
                    st.rerun()
                except Exception as e:
                    st.error("transcribe မအောင်မြင်ပါ")
                    st.code(str(e)[:800])
        with c2:
            st.write("**ကိုယ်ပိုင် transcript တင်မယ်**")
            sj = st.file_uploader("segments JSON", type=["json"], key="segjson")
            if sj:
                try:
                    s2, _ = P.load_project_json(
                        _save_tmp(sj, os.path.join(WD, "up_segs.json")))
                    if s2 is None:
                        st.error("ဒီဖိုင်က ဘာသာပြန်သီးသန့်ပါ — "
                                 "3️⃣ အဆင့်မှာ တင်ပါ")
                    else:
                        st.session_state.segments = s2
                        st.session_state.burmese = None
                        st.session_state.burmese_partial = []
                        st.success(f"{len(s2)} ကြောင်း တင်ပြီးပြီ")
                        st.rerun()
                except Exception as e:
                    st.error(f"ဖိုင်ဖတ်မရပါ: {e}")
    else:
        segs = st.session_state.segments
        st.success(f"စာကြောင်း {len(segs)} ကြောင်း")
        import pandas as pd
        df = pd.DataFrame([{"start": s["start"], "end": s["end"],
                            "text": s["text"]} for s in segs])
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True,
                                height=400, key="seg_editor")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 ပြင်ဆင်ချက်သိမ်းမယ်"):
                new_segs = [
                    {"start": float(r["start"]), "end": float(r["end"]),
                     "text": str(r["text"])}
                    for _, r in edited.iterrows() if str(r["text"]).strip()]
                st.session_state.segments = new_segs
                # keep downstream in sync: drop translation if line count changed
                if (st.session_state.burmese is not None
                        and len(st.session_state.burmese) != len(new_segs)):
                    st.session_state.burmese = None
                    st.session_state.burmese_partial = []
                    st.session_state.groups = None
                    st.session_state.line_mp3s = None
                    st.warning("စာကြောင်းအရေအတွက်ပြောင်းသွားလို့ "
                               "ဘာသာပြန်ကို ပြန်လုပ်ပေးပါ")
                else:
                    st.success("သိမ်းပြီးပြီ!")
        with col2:
            if st.button("🗑️ ပြန်လုပ်မယ် (transcriptဖျက်မယ်)"):
                st.session_state.segments = None
                st.session_state.burmese = None
                st.session_state.burmese_partial = []
                st.session_state.groups = None
                st.session_state.line_mp3s = None
                st.rerun()

# ============================================================ 3. TRANSLATE
elif step.startswith("3"):
    st.header("3️⃣ ဘာသာပြန်ခြင်း (Chinese → မြန်မာ)")
    need_video()
    if st.session_state.segments is None:
        st.warning("အရင် 2️⃣ အဆင့်မှာ transcript ရအောင်လုပ်ပါ")
        st.stop()
    segs = st.session_state.segments

    # ---- translation method: two buttons ----
    if "translate_mode" not in st.session_state:
        st.session_state.translate_mode = "gemini"
    MODES = {
        "apikey": {"label": "🔑 API Key နဲ့ ဘာသာပြန်မယ်",
                   "model": "gpt-4o-mini",
                   "key_help": "platform.openai.com → API keys",
                   "key_hint": "API Key",
                   "url_editable": True},
        "gemini": {"label": "✨ Gemini (အလကား) နဲ့ ဘာသာပြန်မယ်",
                   "model": "gemini-2.0-flash",
                   "key_help": "aistudio.google.com → Get API key (အလကားရတယ်)",
                   "key_hint": "Gemini API Key",
                   "url_editable": False},
    }
    mode = st.session_state.translate_mode
    m1, m2 = st.columns(2)
    for i, m in enumerate(("apikey", "gemini")):
        with (m1 if i == 0 else m2):
            if st.button(MODES[m]["label"],
                         type="primary" if mode == m else "secondary",
                         use_container_width=True, key=f"tmode_{m}"):
                st.session_state.translate_mode = m
                st.rerun()
    cfg = MODES[mode]
    st.caption(f"🔑 Key ထုတ်ရန်: {cfg['key_help']}")
    c1, c2 = st.columns(2)
    with c1:
        if cfg["url_editable"]:
            base_url = st.text_input(
                "API Base URL", value="https://api.openai.com/v1",
                key="base_url_apikey",
                help="OpenAI-compatible API ဆို URL ကို ဒီမှာ ပြင်လို့ရပါတယ်")
        else:
            base_url = "gemini-native"
            st.text_input("API Base URL", value=base_url, disabled=True)
        api_key = st.text_input(cfg["key_hint"], type="password",
                                key=f"api_key_{mode}")
    with c2:
        model = st.text_input("Model", value=cfg["model"],
                              key=f"llm_model_{mode}")
        batch = st.number_input("တစ်ခါတည်း ဘယ်နှစ်ကြောင်းပြောင်းမလဲ",
                                5, 50, 20,
                                help="များလေ request အကြိမ်နည်းလေ")
    delay = st.slider("Request ကြား ခဏစောင့်ချိန် (စက္ကန့်)",
                      0.0, 10.0, 3.0, 0.5,
                      help="429 (limit) မထိအောင် ဖြည်းဖြည်းခေါ်ပါ — "
                           "တက်နေရင် ဒါကို များများထားပါ")
    with st.expander("📝 Prompt ပြင်မယ် (optional)", expanded=False):
        st.caption("Gemini နဲ့ ပြန်ရင် Ko Zay ပြထားတဲ့ JSON prompt ကို "
                   "အလိုအလျောက် သုံးပါတယ် ✅ ကိုယ့် prompt သုံးချင်ရင် "
                   "ဒီမှာ paste လုပ်ပါ — ဗလာထားရင် default သုံးပါမယ်။")
        custom_prompt = st.text_area("System prompt", height=150,
                                     key="custom_prompt")
    partial = st.session_state.get("burmese_partial") or []
    can_resume = bool(api_key and partial and len(partial) < len(segs))
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        test_btn = st.button("🔍 Connection စမ်းမယ်")
    with col_b:
        go_btn = st.button("🌐 ဘာသာပြန်မယ်", type="primary",
                           disabled=not api_key)
    with col_c:
        resume_btn = st.button(f"▶️ ဆက်လုပ် ({len(partial)}/{len(segs)})",
                               disabled=not can_resume,
                               help="ရပ်သွားတဲ့နေရာကနေ ဆက်ဘာသာပြန်မယ် — "
                                    "ပြီးပြီးသားတွေ မပျက်ပါဘူး")
    if test_btn:
        if not api_key:
            st.warning("API Key အရင်ထည့်ပါ")
        else:
            with st.spinner("စမ်းနေပါတယ်…"):
                ok, msg = P.test_llm_connection(base_url, api_key, model)
            (st.success if ok else st.error)(msg)
    if go_btn or resume_btn:
        done = list(partial) if resume_btn else []
        if len(done) >= len(segs):
            done = []
        st.session_state.burmese_partial = list(done)
        bar = st.progress(0.0, "ဘာသာပြန်နေပါတယ်…")

        def _checkpoint(out):
            st.session_state.burmese_partial = out

        try:
            with st.spinner("ဘာသာပြန်နေပါတယ်…"):
                custom = (custom_prompt or "").strip() or None
                if mode == "gemini" and not custom:
                    # Ko Zay's JSON prompt, auto-applied for Gemini
                    sys_prompt = P.TRANSLATE_SYSTEM_JSON
                    in_style = "id_duration"
                else:
                    sys_prompt = custom
                    in_style = "numbered"
                burm = P.translate_llm(
                    segs, api_key, base_url, model, batch,
                    delay_between=delay, done=done,
                    checkpoint_cb=_checkpoint,
                    progress_cb=lambda x: bar.progress(min(x, 1.0)),
                    system=sys_prompt, input_style=in_style)
            st.session_state.burmese = burm
            st.session_state.burmese_partial = []
            bar.progress(1.0, "ပြီးပါပြီ!")
            st.success(f"ပြီးပါပြီ — {len(burm)} ကြောင်း ✅ "
                       "အောက်မှာ စစ်ပြီး ပြင်လို့ရပါတယ်")
            st.rerun()
        except Exception as e:
            msg = str(e)
            done_n = len(st.session_state.get("burmese_partial") or [])
            st.error(f"ဘာသာပြန်မအောင်မြင်ပါ "
                     f"({done_n}/{len(segs)} ကြောင်းပြီးပြီ — မပျက်ပါဘူး)")
            st.code(msg[:800])
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                    or "quota" in msg.lower() or "rate" in msg.lower()):
                st.info("💡 **API limit** ထိသွားတာပါ — ၁-၂ မိနစ်စောင့်ပြီး "
                        "**▶️ ဆက်လုပ်** ကိုနှိပ်ပါ။ ပြီးပြီးသားတွေ "
                        "ဆက်သွားမှာပါ။ ခဏခဏတက်ရင် Request ကြားစောင့်ချိန်ကို "
                        "၅-၁၀ စက္ကန့်ထိထားကြည့်ပါ")

    c1, c2 = st.columns(2)
    with c1:
        pj = st.file_uploader("Project JSON တင်မယ် (transcript.json ပုံစံ)",
                              type=["json"], key="projjson")
        if pj:
            try:
                s2, b2 = P.load_project_json(
                    _save_tmp(pj, os.path.join(WD, "up_proj.json")))
                if s2 is None:  # burmese-lines-only file
                    if len(b2) != len(segs):
                        st.error(f"စာကြောင်းအရေအတွက် မတူပါ "
                                 f"(ဖိုင်: {len(b2)}, transcript: {len(segs)})")
                    else:
                        st.session_state.burmese = b2
                        st.success("ဘာသာပြန်ဖိုင် တင်ပြီးပြီ")
                        st.rerun()
                else:
                    st.session_state.segments = s2
                    st.session_state.burmese = b2
                    st.session_state.burmese_partial = []
                    st.session_state.groups = None
                    st.session_state.line_mp3s = None
                    st.success("project တင်ပြီးပြီ")
                    st.rerun()
            except Exception as e:
                st.error(f"ဖိုင်ဖတ်မရပါ: {e}")
    with c2:
        if st.session_state.burmese:
            P.save_project_json(os.path.join(WD, "project.json"),
                                st.session_state.segments,
                                st.session_state.burmese)
            st.download_button("⬇️ Project JSON ဒေါင်းမယ်",
                               open(os.path.join(WD, "project.json"), "rb"),
                               file_name="project.json")
            st.download_button("⬇️ SRT (မြန်မာစာတန်း) ဒေါင်းမယ်",
                               P.to_srt(st.session_state.segments,
                                        st.session_state.burmese
                                        ).encode("utf-8"),
                               file_name="subtitles_mm.srt",
                               mime="text/plain")

    with st.expander("✍️ အပြင်မှာ ပြန်ထားတဲ့ စာတွေ ထည့်မယ် (paste)", expanded=False):
        st.caption("တရုတ်စာတွေကို အပြင် (ဥပမာ AI chat) မှာ ဘာသာပြန်ပြီး "
                   "မြန်မာစာကြောင်းတွေ ဒီမှာ paste လုပ်လို့ရပါတယ် — "
                   "API limit ပြဿနာ လုံးဝမရှိတော့ဘူး ✅")
        cc1, cc2 = st.columns(2)
        with cc1:
            cn_txt = "\n".join(s["text"] for s in segs)
            st.download_button("⬇️ တရုတ်စာကြောင်းတွေ ဒေါင်းမယ် (.txt)",
                               cn_txt.encode("utf-8"),
                               file_name="chinese_lines.txt",
                               mime="text/plain")
        with cc2:
            st.caption(f"transcript စာကြောင်း: **{len(segs)}** ကြောင်း")
        pasted = st.text_area("မြန်မာဘာသာပြန် (တစ်ကြောင်းချင်းစီ, အစဉ်လိုက်)",
                             height=200, key="paste_mm",
                             placeholder="ပထမစာကြောင်းရဲ့ မြန်မာဘာသာပြန်\n"
                                         "ဒုတိယစာကြောင်းရဲ့ မြန်မာဘာသာပြန်\n…")
        strip_nums = st.checkbox("အစမှာပါတဲ့ နံပါတ်တွေ ဖယ်မယ် (ဥပမာ “1. ”)",
                                value=True)
        import re as _re
        _lines = [(_re.sub(r"^\s*\d+\s*[.)、:]\s*", "", l).strip() if strip_nums
                   else l.strip())
                  for l in (pasted or "").split("\n")]
        _lines = [l for l in _lines if l]
        st.caption(f"တွေ့ရှိစာကြောင်း: **{len(_lines)}** / လိုအပ်တာ: **{len(segs)}**")
        if st.button("✅ ဒီစာတွေသုံးမယ်", type="primary"):
            if len(_lines) != len(segs):
                st.error(f"အရေအတွက် မကိုက်ဘူး — စာကြောင်း {len(segs)} ကြောင်း "
                         f"လိုတယ်, {len(_lines)} ကြောင်း တွေ့တယ်။ "
                         f"စစ်ကြည့်ပါ")
            else:
                st.session_state.burmese = _lines
                st.session_state.burmese_partial = []
                st.session_state.groups = None
                st.session_state.line_mp3s = None
                st.session_state.line_durs = None
                st.session_state.final_path = None
                st.success("ထည့်ပြီးပြီ! ✅ အောက်မှာ စစ်ပြီး ပြင်လို့ရပါတယ်")
                st.rerun()

    if st.session_state.burmese:
        st.subheader("စစ်ဆေးပြင်ဆင်ရန်")
        import pandas as pd
        df = pd.DataFrame([{"original": s["text"], "burmese": b}
                           for s, b in zip(segs, st.session_state.burmese)])
        edited = st.data_editor(df, use_container_width=True, height=450,
                                key="burm_editor",
                                column_config={"original": st.column_config.TextColumn(
                                    "Chinese", disabled=True)})
        if st.button("💾 ဘာသာပြန်သိမ်းမယ်"):
            st.session_state.burmese = [str(r["burmese"]) for _, r in edited.iterrows()]
            if st.session_state.line_mp3s:
                st.session_state.line_mp3s = None
                st.session_state.groups = None
                st.warning("စာသားပြောင်းသွားလို့ အသံကို 4️⃣ အဆင့်မှာ ပြန်ထုတ်ပေးပါ")
            else:
                st.success("သိမ်းပြီးပြီ!")
    else:
        st.info("👆 အပေါ်က auto-translate လုပ်ပါ၊ ဒါမှမဟုတ် project JSON တင်ပါ")

# ============================================================ 4. TTS
elif step.startswith("4"):
    st.header("4️⃣ အသံသွင်းခြင်း (Burmese voiceover)")
    need_video()
    if st.session_state.burmese is None:
        st.warning("အရင် 3️⃣ အဆင့်မှာ ဘာသာပြန်အပြီးလုပ်ပါ")
        st.stop()
    segs, burm = st.session_state.segments, st.session_state.burmese

    c1, c2, c3 = st.columns(3)
    with c1:
        voice_name = st.selectbox("အသံ", list(VOICES.keys()))
    with c2:
        speed = st.slider("အမြန်နှုန်း (%)", -20, 40, 0,
                          help="Edge TTS အသံအမြန်နှုန်း — စကားပြောမြန်တဲ့ "
                               "ဗီဒီယိုဆို +20~+30 ထားကြည့်ပါ")
    with c3:
        n_groups = st.slider("Group အရေအတွက်", 2, 8, 4,
                             help="နားထောင်ကြည့်ရန် အပိုင်းခွဲတာ — "
                                  "sync က စာတစ်ကြောင်းချင်းစီ သူ့ timestamp "
                                  "အတိုင်းမို့ group နဲ့ မဆိုင်ပါ")

    if st.button("📐 Group ခွဲမယ်", type="primary"):
        st.session_state.groups = P.balance_groups(segs, burm, n_groups)
        st.session_state.line_mp3s = None
        st.session_state.final_path = None
        st.rerun()

    if st.session_state.groups:
        groups = st.session_state.groups
        st.subheader("Group များ (ခန့်မှန်းအမြန်နှုန်း)")
        import pandas as pd
        rows = []
        for i, (a, b) in enumerate(groups):
            span = segs[b]["end"] - segs[a]["start"]
            est = P.estimate_narration_secs(burm[a:b + 1])
            rows.append({"group": i + 1, "segments": f"{a}–{b}",
                         "count": b - a + 1,
                         "video span (s)": round(span, 1),
                         "est narration (s)": round(est, 1),
                         "est rate": round(est / span, 2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        worst = max(r["est rate"] for r in rows)
        if worst > P.MAX_TEMPO:
            st.error(f"⚠️ Group တစ်ခုမှာ {worst:.2f}x လိုနေတယ် "
                     f"({P.MAX_TEMPO}x ထက်များရင် အသံအရမ်းမြန်မယ်) — "
                     f"group အရေအတွက်တိုးကြည့်ပါ")
        elif worst > 1.4:
            st.warning(f"အမြန်ဆုံး group က {worst:.2f}x — "
                       "နည်းနည်းမြန်မယ်၊ group တိုးလို့ရပါတယ်")
        else:
            st.success("အမြန်နှုန်းတွေ အဆင်ပြေပါတယ် 👍")

        if st.button("🎙️ အသံထုတ်မယ် (Edge TTS)", type="primary"):
            bar = st.progress(0.0, "အသံထုတ်နေပါတယ်…")
            try:
                tts_dir = os.path.join(WD, "lines")
                mp3s = P.synthesize_lines(
                    burm, VOICES[voice_name], speed, tts_dir,
                    progress_cb=lambda x: bar.progress(
                        min(x, 1.0), f"အသံထုတ်နေပါတယ်… "
                                    f"{int(x * len(burm))}/{len(burm)} ကြောင်း"))
                # measure real durations for the sync report
                durs = [P.media_duration(p) for p in mp3s]

                def _avail(j):
                    s = segs[j]
                    slot = max(s["end"] - s["start"], 0.01)
                    nxt = segs[j + 1]["start"] if j + 1 < len(segs) \
                        else s["start"] + slot
                    return max(nxt - s["start"], slot)

                # overflow fix-up: lines that still don't fit even at max
                # tempo are re-synthesized at a faster TTS voice
                bad = [j for j, d in enumerate(durs)
                       if d / _avail(j) > P.MAX_TEMPO]
                if bad:
                    fast_speed = min(speed + 30, 50)
                    st.info(f"⏩ စာကြောင်း {len(bad)} ကြောင်း အချိန်မလောက်လို့ "
                            f"အသံနှုန်း {fast_speed:+d}% နဲ့ ပြန်ထုတ်ပေးမယ်…")
                    P.synthesize_selected(
                        bad, burm, VOICES[voice_name], fast_speed, mp3s,
                        progress_cb=lambda x: bar.progress(
                            min(x, 1.0),
                            f"ပြန်ထုတ်နေပါတယ်… {int(x * len(bad))}/{len(bad)}"))
                    durs = [P.media_duration(p) for p in mp3s]
                st.session_state.line_mp3s = mp3s
                st.session_state.line_durs = durs
                # group preview mp3s (for listening only)
                for i, (a, b) in enumerate(groups):
                    pv = os.path.join(WD, f"preview_{i}.mp3")
                    lst = os.path.join(WD, f"preview_{i}.txt")
                    with open(lst, "w") as f:
                        for p in mp3s[a:b + 1]:
                            f.write(f"file '{p}'\n")
                    P.run([P.FFMPEG, "-y", "-v", "error", "-f", "concat",
                           "-safe", "0", "-i", lst, "-c", "copy", pv])
                # sync report: which lines fit, which will be cut
                still = [j for j, d in enumerate(durs)
                         if d / _avail(j) > P.MAX_TEMPO]
                if still:
                    nums = ", ".join(f"#{j+1}" for j in still[:8])
                    more = f" (+{len(still)-8} more)" if len(still) > 8 else ""
                    st.warning(f"⚠️ စာကြောင်း {len(still)} ကြောင်း ({nums}{more}) "
                               f"အချိန်အရမ်းမလောက်ဘူး — 5️⃣ မှာ ဖြတ်တောက်ခံရမယ် "
                               f"(အသံထပ်တာတော့ မဖြစ်တော့ဘူး ✅)။ "
                               f"စာကို တိုအောင်ပြင်တာ အကောင်းဆုံးပါ")
                else:
                    n_sped = sum(1 for j, d in enumerate(durs)
                                 if d / max(segs[j]["end"] - segs[j]["start"], 0.01) > 1.005)
                    st.success(f"အဆင်ပြေပါပြီ 👍 စာကြောင်း {n_sped} ကြောင်း "
                               f"အနည်းငယ်မြန်ပေးထားတယ်၊ ထပ်နေတာမရှိဘူး ✅")
                st.rerun()
            except Exception as e:
                st.error("TTS မအောင်မြင်ပါ — internet / Edge TTS စစ်ကြည့်ပါ")
                st.code(str(e)[:800])

    if st.session_state.line_mp3s:
        st.divider()
        st.subheader("🔊 Group အသံများ (နားထောင်ရန်)")
        for i, (a, b) in enumerate(st.session_state.groups or []):
            pv = os.path.join(WD, f"preview_{i}.mp3")
            if os.path.exists(pv):
                st.write(f"Group {i+1} (စာကြောင်း {a+1}–{b+1})")
                st.audio(pv)
        if st.button("🔄 အသံပြန်ထုတ်မယ်"):
            st.session_state.line_mp3s = None
            st.session_state.groups = None
            st.rerun()

# ============================================================ 5. ASSEMBLE
elif step.startswith("5"):
    st.header("5️⃣ ပေါင်းစပ်ခြင်း")
    need_video()
    if not st.session_state.line_mp3s:
        st.warning("အရင် 4️⃣ အဆင့်မှာ အသံထုတ်ပြီးမှ ပေါင်းလို့ရပါမယ်")
        st.stop()

    mode = st.radio("ပေါင်းစပ်နည်း",
        ["🎯 အသံကို video နဲ့ကိုက် (အသံမြန်ပေးမယ်)",
         "🎬 video ကို အသံနဲ့ကိုက် (freeze နဲ့ရှည်ပေးမယ်)",
         "🐢 video ကို အသံနဲ့ကိုက် (slow-mo နှေးပြီးရှည်ပေးမယ်)"],
        help="🎯: video အရှည်မပြောင်း၊ အသံမြန်မယ်။ "
             "🎬: အသံသဘာဝအတိုင်း၊ video ရပ်ပြီး ရှည်မယ်။ "
             "🐢: အသံသဘာဝအတိုင်း၊ video နှေးပြီး ရှည်မယ် (ပိုသဘာဝကျတယ်)")
    slowmo = mode.startswith("🐢")
    freeze = mode.startswith("🎬")

    if slowmo:
        maxslow = st.slider("video အနှေးဆုံး", 1.2, 2.0, 1.5, 0.1,
                            help="ဒီထက်ပိုနှေးစရာမလိုဘူး — ကျန်တာကို "
                                 "freeze နဲ့ဖြည့်ပေးမယ်")
        duck = st.slider("မူရင်းအသံ အတိုးအကျယ် (%)", 0, 30, 12,
                         help="မူရင်းအသံကိုလည်း video နဲ့အတူ နှေးပြီး "
                              "တိုးတိုးလေးထားမယ်") / 100
        st.info("ℹ️ 🐢 mode: အသံ သဘာဝအတိုင်း (မမြန်ဘူး) — စကားရှည်တဲ့နေရာ "
                "video နှေးသွားမယ်")
        ftempo = None
    elif freeze:
        ftempo = st.slider("အသံအမြန်နှုန်း အများဆုံး", 1.0, 2.0, 1.3, 0.05,
                           help="ဒီထက်ပိုမြန်စရာမလိုဘူး — ကျန်တာကို "
                                "video freeze နဲ့ ဖြည့်ပေးမယ်")
        st.info("ℹ️ freeze mode မှာ မူရင်းအသံကို ပိတ်ထားမယ် "
                "(ရပ်ထားတဲ့အချိန်တွေနဲ့ မကိုက်လို့)")
        duck = 0.0
        maxslow = None
    else:
        duck = st.slider("မူရင်းအသံ အတိုးအကျယ် (%)", 0, 30, 12,
                         help="နောက်ခံမူရင်းအသံ ဘယ်လောက်ထားမလဲ") / 100
        ftempo = None
        maxslow = None

    if st.button("🎬 Final video ထုတ်မယ်", type="primary"):
        out = os.path.join(WD, "dubbed_final.mp4")
        bar = st.progress(0.0, "ပေါင်းစပ်နေပါတယ်…")
        try:
            if slowmo:
                res = P.assemble_slowmo(
                    st.session_state.video_path, st.session_state.segments,
                    st.session_state.line_mp3s, out, max_slow=maxslow,
                    duck=duck,
                    progress_cb=lambda x: bar.progress(min(x, 1.0)))
                st.session_state.srt_segments = res["new_segments"]
            elif freeze:
                res = P.assemble_freeze(
                    st.session_state.video_path, st.session_state.segments,
                    st.session_state.line_mp3s, out, max_tempo=ftempo,
                    progress_cb=lambda x: bar.progress(min(x, 1.0)))
                st.session_state.srt_segments = res["new_segments"]
            else:
                res = P.assemble(
                    st.session_state.video_path, st.session_state.segments,
                    st.session_state.line_mp3s, out,
                    duck=duck,
                    progress_cb=lambda x: bar.progress(min(x, 1.0)))
                st.session_state.srt_segments = st.session_state.segments
            res["freeze"] = freeze
            res["slowmo"] = slowmo
            st.session_state.final_path = out
            st.session_state.final_info = res
            bar.progress(1.0)
            st.rerun()
        except Exception as e:
            st.error("ပေါင်းစပ်မအောင်မြင်ပါ")
            st.code(str(e)[:1200])

    if st.session_state.final_path:
        info = st.session_state.final_info
        if info.get("slowmo"):
            st.success(f"ပြီးပြီ! 🎉 ({info['new_duration']:.0f} စက္ကန့် — "
                       f"{info['added_sec']:.0f}s ရှည်သွားတယ်, "
                       f"{info['slowmos']} နေရာ slow-mo, "
                       f"{info['extensions']} နေရာ freeze)")
        elif info.get("freeze"):
            st.success(f"ပြီးပြီ! 🎉 ({info['new_duration']:.0f} စက္ကန့် — "
                       f"{info['added_sec']:.0f}s ရှည်သွားတယ်, "
                       f"{info['extensions']} နေရာမှာ freeze)")
        else:
            st.success(f"ပြီးပြီ! 🎉 ({info['duration']:.0f} စက္ကန့်)")
        if info.get("slowmo"):
            sf = info["slow_factors"]
            st.write(f"video အနှေးနှုန်း — အနှေးဆုံး: {max(sf):.2f}x, "
                     f"ပျမ်းမျှ: {sum(sf)/len(sf):.2f}x "
                     f"(အသံကတော့ သဘာဝအတိုင်း 1.00x)")
        else:
            rates = info["rates"]
            st.write(f"စာကြောင်းအမြန်နှုန်း — အမြန်ဆုံး: {max(rates):.2f}x, "
                     f"ပျမ်းမျှ: {sum(rates)/len(rates):.2f}x")
        for w in info["warnings"][:5]:
            st.warning(w)
        if len(info["warnings"]) > 5:
            st.caption(f"...နောက်ထပ် {len(info['warnings'])-5} ခု")
        st.video(st.session_state.final_path)
        with open(st.session_state.final_path, "rb") as f:
            st.download_button("⬇️ Dubbed MP4 ဒေါင်းလုဒ်ဆွဲမယ်", f,
                               file_name="dubbed_final.mp4",
                               mime="video/mp4", type="primary")
        st.download_button("⬇️ SRT (မြန်မာစာတန်း) ဒေါင်းမယ်",
                           P.to_srt(st.session_state.get("srt_segments")
                                    or st.session_state.segments,
                                    st.session_state.burmese).encode("utf-8"),
                           file_name="subtitles_mm.srt", mime="text/plain")
