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
ss("groups", None)        # [(a,b)]
ss("group_mp3s", None)    # [path]
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
                    st.session_state.groups = None
                    st.session_state.group_mp3s = None
                    st.warning("စာကြောင်းအရေအတွက်ပြောင်းသွားလို့ "
                               "ဘာသာပြန်ကို ပြန်လုပ်ပေးပါ")
                else:
                    st.success("သိမ်းပြီးပြီ!")
        with col2:
            if st.button("🗑️ ပြန်လုပ်မယ် (transcriptဖျက်မယ်)"):
                st.session_state.segments = None
                st.session_state.burmese = None
                st.session_state.groups = None
                st.session_state.group_mp3s = None
                st.rerun()

# ============================================================ 3. TRANSLATE
elif step.startswith("3"):
    st.header("3️⃣ ဘာသာပြန်ခြင်း (Chinese → မြန်မာ)")
    need_video()
    if st.session_state.segments is None:
        st.warning("အရင် 2️⃣ အဆင့်မှာ transcript ရအောင်လုပ်ပါ")
        st.stop()
    segs = st.session_state.segments

    with st.expander("🤖 Auto-translate (LLM API)", expanded=st.session_state.burmese is None):
        st.caption("ဘာသာပြန်အတွက် LLM API key လိုပါတယ်")
        PRESETS = {
            "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini",
                       "platform.openai.com → API keys"),
            "Gemini (အလကား)": ("https://generativelanguage.googleapis.com/v1beta/openai/",
                                "gemini-2.0-flash",
                                "aistudio.google.com → Get API key (အလကားရတယ်)"),
            "Custom": ("", "", ""),
        }
        if "base_url" not in st.session_state:
            st.session_state.base_url = PRESETS["OpenAI"][0]
            st.session_state.llm_model = PRESETS["OpenAI"][1]
            st.session_state.provider_applied = "OpenAI"
        provider = st.selectbox("Provider", list(PRESETS.keys()),
                                key="provider")
        if provider != st.session_state.get("provider_applied"):
            st.session_state.base_url = PRESETS[provider][0]
            st.session_state.llm_model = PRESETS[provider][1]
            st.session_state.provider_applied = provider
            st.rerun()
        st.caption(f"🔑 Key ထုတ်ရန်: {PRESETS[provider][2]}")
        c1, c2 = st.columns(2)
        with c1:
            base_url = st.text_input("API Base URL", key="base_url")
            api_key = st.text_input("API Key", type="password", key="api_key")
        with c2:
            model = st.text_input("Model", key="llm_model")
            batch = st.number_input("တစ်ခါတည်း ဘယ်နှစ်ကြောင်းပြောင်းမလဲ",
                                    5, 50, 20,
                                    help="များလေ request အကြိမ်နည်းလေ — "
                                         "429 (Too Many Requests) တက်ရင် "
                                         "20–30 လောက်ထားပါ")
        col_a, col_b = st.columns(2)
        with col_a:
            test_btn = st.button("🔍 Connection စမ်းမယ်")
        with col_b:
            go_btn = st.button("🌐 ဘာသာပြန်မယ်", type="primary",
                               disabled=not api_key)
        if test_btn:
            if not api_key:
                st.warning("API Key အရင်ထည့်ပါ")
            else:
                with st.spinner("စမ်းနေပါတယ်…"):
                    ok, msg = P.test_llm_connection(base_url, api_key, model)
                (st.success if ok else st.error)(msg)
        if go_btn:
            bar = st.progress(0.0, "ဘာသာပြန်နေပါတယ်…")
            try:
                burm = P.translate_llm(
                    segs, api_key, base_url, model, batch,
                    progress_cb=lambda x: bar.progress(min(x, 1.0)))
                st.session_state.burmese = burm
                bar.progress(1.0)
                st.success("ပြီးပြီ! အောက်မှာ စစ်ပြီး ပြင်လို့ရပါတယ်")
                st.rerun()
            except Exception as e:
                msg = str(e)
                st.error("ဘာသာပြန်မအောင်မြင်ပါ")
                st.code(msg[:800])
                if "429" in msg:
                    st.info("💡 **429 = API rate limit** ပြည့်သွားတာပါ — "
                            "အခု auto-retry ပါပြီးသားမို့ ခဏစောင့်ပြီး "
                            "🌐 ဘာသာပြန်မယ် ကို ပြန်နှိပ်ကြည့်ပါ။ "
                            "တစ်ခါတည်း အကြောင်းရေ **20–30** လောက်ထားရင် "
                            "request အကြိမ်နည်းပြီး ပိုအဆင်ပြေပါတယ်။ "
                            "ခဏခဏတက်နေရင် OpenAI dashboard မှာ usage/limit "
                            "စစ်ကြည့်ပါ")

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
                    st.session_state.groups = None
                    st.session_state.group_mp3s = None
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
        speed = st.slider("အမြန်နှုန်း (%)", -20, 20, 0,
                          help="Edge TTS အသံအမြန်နှုန်း")
    with c3:
        n_groups = st.slider("Group အရေအတွက်", 2, 8, 4,
                             help="များလေ အသံသဘာဝကျလေ၊ TTS ခေါ်တာများလေပေါ့")

    if st.button("📐 Group ခွဲမယ်", type="primary"):
        st.session_state.groups = P.balance_groups(segs, burm, n_groups)
        st.session_state.group_mp3s = None
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
            mp3s, bar = [], st.progress(0.0, "အသံထုတ်နေပါတယ်…")
            try:
                for i, (a, b) in enumerate(groups):
                    text = " ".join(burm[a:b + 1]).strip()
                    p = os.path.join(WD, f"group_{i}.mp3")
                    P.synthesize_group(text, VOICES[voice_name], speed, p)
                    mp3s.append(p)
                    bar.progress((i + 1) / len(groups))
                st.session_state.group_mp3s = mp3s
                st.success("အသံရပြီ! နားထောင်ကြည့်ပါ 👇")
                st.rerun()
            except Exception as e:
                st.error("TTS မအောင်မြင်ပါ — internet / Edge TTS စစ်ကြည့်ပါ")
                st.code(str(e)[:800])

    if st.session_state.group_mp3s:
        st.divider()
        st.subheader("🔊 Group အသံများ")
        for i, p in enumerate(st.session_state.group_mp3s):
            st.write(f"Group {i+1}")
            st.audio(p)
        if st.button("🔄 အသံပြန်ထုတ်မယ်"):
            st.session_state.group_mp3s = None
            st.session_state.groups = None
            st.rerun()

# ============================================================ 5. ASSEMBLE
elif step.startswith("5"):
    st.header("5️⃣ ပေါင်းစပ်ခြင်း")
    need_video()
    if not st.session_state.group_mp3s:
        st.warning("အရင် 4️⃣ အဆင့်မှာ အသံထုတ်ပြီးမှ ပေါင်းလို့ရပါမယ်")
        st.stop()

    duck = st.slider("မူရင်းအသံ အတိုးအကျယ် (%)", 0, 30, 12,
                     help="နောက်ခံမူရင်းအသံ ဘယ်လောက်ထားမလဲ")
    if st.button("🎬 Final video ထုတ်မယ်", type="primary"):
        out = os.path.join(WD, "dubbed_final.mp4")
        bar = st.progress(0.0, "ပေါင်းစပ်နေပါတယ်…")
        try:
            res = P.assemble(
                st.session_state.video_path, st.session_state.segments,
                st.session_state.groups, st.session_state.group_mp3s, out,
                duck=duck / 100,
                progress_cb=lambda x: bar.progress(min(x, 1.0)))
            st.session_state.final_path = out
            st.session_state.final_info = res
            bar.progress(1.0)
            st.rerun()
        except Exception as e:
            st.error("ပေါင်းစပ်မအောင်မြင်ပါ")
            st.code(str(e)[:1200])

    if st.session_state.final_path:
        info = st.session_state.final_info
        st.success(f"ပြီးပြီ! 🎉 ({info['duration']:.0f} စက္ကန့်)")
        st.write("Group အမြန်နှုန်းများ:", info["rates"])
        for w in info["warnings"]:
            st.warning(w)
        st.video(st.session_state.final_path)
        with open(st.session_state.final_path, "rb") as f:
            st.download_button("⬇️ Dubbed MP4 ဒေါင်းလုဒ်ဆွဲမယ်", f,
                               file_name="dubbed_final.mp4",
                               mime="video/mp4", type="primary")
