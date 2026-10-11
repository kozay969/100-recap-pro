import os
import json
import streamlit as st
import assemblyai as aai
from google import genai
from google.genai import types
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip
from utils.subtitle_utils import validate_subtitle_timing

# Page Setup
st.set_page_config(
    page_title="Advanced AI Movie Recap Pipeline",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Advanced AI Movie Recap Tool (AssemblyAI + Gemini API)")
st.markdown("ဗီဒီယိုမှ စာသားထုတ်ခြင်း၊ Gemini API ဖြင့် တိုက်ရိုက်ဘာသာပြန်ဆိုခြင်းနှင့် အသံထုတ်လုပ်ခြင်းများကို တစ်နေရာတည်းတွင် လုပ်ဆောင်ပါ။")

# Sidebar for API Keys
st.sidebar.header("🔑 API Configuration")
assemblyai_api_key = st.sidebar.text_input("AssemblyAI API Key", type="password")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

if assemblyai_api_key:
    aai.settings.api_key = assemblyai_api_key

# Tabs Layout
tab1, tab2, tab3 = st.tabs(["1. Video & Transcript", "2. Gemini Direct Translation", "3. Audio & Video Sync"])

with tab1:
    st.header("Step 1: ဗီဒီယိုတင်ခြင်းနှင့် စာသားထုတ်ခြင်း")
    uploaded_video = st.file_uploader("ဗီဒီယိုဖိုင်ကို တင်ပါ (MP4, MKV)", type=["mp4", "mkv", "avi"])
    
    if uploaded_video is not None:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.video(video_path)
        
        if st.button("🚀 AssemblyAI ဖြင့် စာသားနှင့် Timing ထုတ်မည်"):
            if not assemblyai_api_key:
                st.error("ကျေးဇူးပြု၍ Sidebar တွင် AssemblyAI API Key ထည့်ပါ။")
            else:
                with st.spinner("AssemblyAI ဖြင့် အသံများကို စာသားပြောင်းနေပါပြီ..."):
                    try:
                        transcriber = aai.Transcriber()
                        transcript = transcriber.transcribe(video_path)
                        
                        if transcript.status == aai.TranscriptStatus.error:
                            st.error(f"Error: {transcript.error}")
                        else:
                            st.success("စာသားထုတ်ယူမှု အောင်မြင်ပါသည်။")
                            st.session_state['transcript_text'] = transcript.text
                            
                            # Utterances or segments if available
                            segments = [{"start": u.start, "end": u.end, "text": u.text} for u in transcript.utterances] if transcript.utterances else []
                            st.session_state['segments'] = segments
                            
                            # JSON format for backup or prompt
                            payload = {"full_text": transcript.text, "segments": segments}
                            st.session_state['transcript_json'] = json.dumps(payload, ensure_ascii=False, indent=4)
                            
                    except Exception as e:
                        st.error(f"အမှားအယွင်း ဖြစ်ပေါ်သည်: {e}")

    if 'transcript_text' in st.session_state:
        st.subheader("📝 ထွက်လာသော မူရင်းစာသား")
        st.text_area("Transcript", st.session_state['transcript_text'], height=150)
        
        if 'segments' in st.session_state and st.session_state['segments']:
            st.subheader("⏱️ Subtitle Timing စစ်ဆေးချက်များ")
            timing_issues = validate_subtitle_timing(st.session_state['segments'])
            if timing_issues:
                for issue in timing_issues[:5]:
                    st.warning(issue)
            else:
                st.info("Subtitle အချိန်ကိုက်မှာ သင့်လျော်မှု ရှိပါသည်။")

with tab2:
    st.header("Step 2: Gemini API ဖြင့် တိုက်ရိုက်ဘာသာပြန်ဆိုခြင်း")
    
    recap_tone = st.selectbox(
        "Recap Narration ပုံစံရွေးချယ်ရန်",
        ["စိတ်လှုပ်ရှားဖွယ်ရာ (Exciting & Dramatic)", "ပြေပြစ်ရှင်းလືင်းသော (Smooth & Clear)", "ဟာသနှောသော (Humorous)"]
    )
    
    if st.button("✨ Gemini API ဖြင့် မြန်မာဘာသာသို့ ဘာသာပြန်မည်"):
        if not gemini_api_key:
            st.error("ကျေးဇူးပြု၍ Sidebar တွင် Gemini API Key ထည့်ပါ။")
        elif 'transcript_text' not in st.session_state:
            st.warning("ကျေးဇူးပြု၍ ပထမအဆင့်တွင် စာသားများ အရင်ထုတ်ပါ။")
        else:
            with st.spinner("Gemini API မှ ဇာတ်ကြောင်းပြန် (Movie Recap) ပုံစံဖြင့် ဘာသာပြန်နေပါပြီ..."):
                try:
                    # Initialize Gemini Client using official google-genai SDK
                    client = genai.Client(api_key=gemini_api_key)
                    
                    prompt = f"""
                    သင်သည် ပရော်ဖက်ရှင်နယ် Movie Recap ရေးသားသူတစ်ဦး ဖြစ်ပါသည်။ 
                    အောက်ပါ မူရင်းအင်္ဂလိပ်စာသားများကို ပရိသတ်စိတ်ဝင်စားစေရန် '{recap_tone}' ပုံစံဖြင့် မြန်မာဘာသာသို့ ဇာတ်ကြောင်းပြန်အဖြစ် ဘာသာပြန်ဆိုပေးပါ။ 
                    ရလဒ်ကို မြန်မာဘာသာ သက်သက်ဖြင့်သာ တိုက်ရိုက်ထုတ်ပေးပါ။
                    
                    မူရင်းစာသား:
                    {st.session_state['transcript_text']}
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    
                    st.session_state['translated_text'] = response.text
                    st.success("Gemini ဖြင့် ဘာသာပြန်ဆိုမှု အောင်မြင်ပါသည်။")
                    
                except Exception as e:
                    st.error(f"Gemini API ချိတ်ဆက်ရာတွင် အမှားဖြစ်ပေါ်သည်: {e}")

    if 'translated_text' in st.session_state:
        st.subheader("🇲🇲 ဘာသာပြန်ပြီးသား မြန်မာဇာတ်ကြောင်းစာသား")
        edited_translation = st.text_area("Edit Translation", st.session_state['translated_text'], height=200)
        st.session_state['translated_text'] = edited_translation

with tab3:
    st.header("Step 3: အသံထုတ်ခြင်းနှင့် ဗီဒီယိုနှင့်ကိုက်ညီအောင် ပေါင်းစပ်ခြင်း")
    
    if 'translated_text' in st.session_state and st.session_state['translated_text']:
        if st.button("🔊 gTTS ဖြင့် မြန်မာအသံဖိုင် ထုတ်မည်"):
            with st.spinner("အသံဖိုင် ဖန်တီးနေပါပြီ..."):
                tts = gTTS(text=st.session_state['translated_text'], lang='my', slow=False)
                audio_path = "output_audio.mp3"
                tts.save(audio_path)
                st.session_state['audio_path'] = audio_path
                st.success("အသံဖိုင် အောင်မြင်စွာ ထွက်ရှိလာပါပြီ။")
                
        if 'audio_path' in st.session_state and os.path.exists('temp_video.mp4'):
            st.audio(st.session_state['audio_path'])
            
            if st.button("🎬 ဗီဒီယိုနှင့် အသံကို ကိုက်ညီအောင် ပေါင်းစပ်မည်"):
                with st.spinner("ဗီဒီယိုနှင့် အသံကို ပေါင်းစပ်နေပါပြီ..."):
                    try:
                        video_clip = VideoFileClip("temp_video.mp4")
                        audio_clip = AudioFileClip(st.session_state['audio_path'])
                        
                        final_clip = video_clip.set_audio(audio_clip)
                        output_video_path = "final_recap_video.mp4"
                        final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
                        
                        st.success("Final Recap Video ထွက်ရှိပါပြီ!")
                        st.video(output_video_path)
                        
                        with open(output_video_path, "rb") as file:
                            st.download_button(
                                label="📥 Final Recap Video ကို Download ရယူမည်",
                                data=file,
                                file_name="movie_recap_output.mp4",
                                mime="video/mp4"
                            )
                    except Exception as e:
                        st.error(f"ဗီဒီယိုပေါင်းစပ်ရာတွင် အမှားဖြစ်ပေါ်သည်: {e}")
    else:
        st.warning("ကျေးဇူးပြု၍ Step 2 တွင် ဘာသာပြန်စာသားများ အရင်ရယူပါ။")
           
