🎬 Dubbing Studio
Rednote (Xiaohongshu) ဗီဒီယို link ထည့်လိုက်တာနဲ့ မြန်မာ dubbing ဗီဒီယို
အချောထုတ်ပေးတဲ့ web app — Streamlit ပေါ်မှာ run ပါတယ်။
Pipeline: ဗီဒီယို → စာသားခွဲခြင်း (Whisper, Chinese) → ဘာသာပြန်ခြင်း
(LLM, မြန်မာ) → အသံသွင်းခြင်း (Edge TTS, Thiha/Nilar) →
အချိန်ကိုက်ပေါင်းစပ်ခြင်း → Final MP4
အဆင့်တိုင်းမှာ စစ်ဆေး/ပြင်ဆင်လို့ရသလို၊ ကိုယ်ပိုင်ဖိုင် (ဗီဒီယို / transcript /
project JSON) တင်ပြီး အဆင့်ကျော်လို့လည်းရပါတယ်။

🚀 Streamlit Cloud ပေါ် တင်နည်း
GitHub repo အသစ်ဖန်တီးပါ (public)
ဒီဖိုင် ၃ ခု ကို repo root မှာ တင်ပါ
app.py
pipeline.py
requirements.txt
share.streamlit.io → New app →
   repo ရွေးပါ → Deploy
ပထမအကြိမ် deploy က ၅–၁၀ မိနစ်ကြာနိုင်ပါတယ်
   (faster-whisper, torch စတဲ့ library တွေကြောင့်)
💡 API key တွေကို code ထဲမှာ ထည့်စရာမလိုပါ — app ရဲ့ UI မှာပဲ
ရိုက်ထည့်ရပါတယ် (session တစ်ခုစာပဲ မှတ်ထားပါတယ်)။
📝 သုံးနည်း
အဆင့်
လုပ်စရာ
1️⃣ ဗီဒီယို
Rednote/xhslink ထည့်ပြီး ဒေါင်းမယ် — login တောင်းတဲ့ link ဆို cookies.txt တင်ပေးပါ။ ဒါမှမဟုတ် MP4 ဖိုင် တိုက်ရိုက်တင်လို့ရပါတယ်
2️⃣ စာသားခွဲခြင်း
Whisper model ရွေး → Transcribe။ ကိုယ်ပိုင် transcript JSON လည်း တင်လို့ရပါတယ်
3️⃣ ဘာသာပြန်ခြင်း
ခလုတ် ၂ ခု: 🔑 API Key (OpenAI-compatible, URL ပြင်လို့ရ) / ✨ Gemini (အလကား) — Gemini ရွေးရင် Ko Zay ရဲ့ JSON prompt ({"translations": [...]}) ကို အလိုအလျောက် သုံးပါတယ်။ Custom prompt လည်း ထည့်လို့ရ။ ပြီးရင် table မှာ ပြင်လို့ရ၊ Project JSON ထုတ်/သွင်းလို့ရ၊ SRT ဒေါင်းလို့ရ
4️⃣ အသံသွင်းခြင်း
အသံ (Thiha/Nilar)၊ အမြန်နှုန်း ရွေး → အသံထုတ် (စာတစ်ကြောင်းချင်းစီ သူ့ timestamp အတိုင်း ကိုက်မယ်) → နားထောင်စစ်ပါ
5️⃣ ပေါင်းစပ်ခြင်း
နည်းလမ်းသုံးမျိုးရွေးလို့ရပါတယ် — 🎯 အသံကို video နဲ့ကိုက် (အသံမြန်ပေးမယ်၊ video အရှည်မပြောင်း) / 🎬 video ကို အသံနဲ့ကိုက် (freeze) (အသံကို သဘာဝအတိုင်းထား၊ လိုအပ်တဲ့နေရာ video ကို freeze နဲ့ရှည်ပေးမယ် — video ရှည်သွားမယ်၊ မူရင်းအသံပိတ်ထားမယ်) / 🐢 video ကို အသံနဲ့ကိုက် (slow-mo) (အသံကို သဘာဝအတိုင်းထား၊ လိုအပ်တဲ့နေရာ video ကို slow motion နှေးပြီးရှည်ပေးမယ် — freeze ထက် ပိုသဘာဝကျတယ်၊ မူရင်းအသံကိုလည်း နှေးပြီး တိုးတိုးထားမယ်)။ MP4 + SRT ဒေါင်းလုဒ်ဆွဲပါ
🔑 cookies.txt ထုတ်နည်း (Rednote login link အတွက်)
Browser မှာ xiaohongshu.com ကို login ဝင်ပါ
Get cookies.txt LOCALLY extension သွင်းပါ (Chrome/Edge)
xiaohongshu.com tab မှာ extension ကိုနှိပ် → Export → cookies.txt ရပါမယ်
App ရဲ့ 1️⃣ အဆင့်မှာ အဲဒီဖိုင်ကို တင်ပေးပါ
⚠️ သိထားရန်
429 Too Many Requests တက်ရင် API rate limit ပြည့်တာပါ — app က
  သူ့အလိုလို retry လုပ်ပေးပါတယ်။ တစ်ခါတည်း အကြောင်းရေ 20–30 ထားပြီး
  ခဏစောင့်ပြန်နှိပ်ပါ။ ခဏခဏဖြစ်ရင် OpenAI dashboard မှာ usage/limit စစ်ပါ
ပထမ transcribe အကြိမ်မှာ Whisper model (~500MB) ဒေါင်းရလို့ ကြာပါမယ်
Streamlit Cloud free plan (RAM 1GB) မှာ small model အဆင်မပြေရင်
  base/tiny သုံးပါ၊ ဒါမှမဟုတ် transcript JSON အဆင်သင့်တင်ပါ
ဘာသာပြန်အတွက် LLM API key လိုပါတယ် — Provider ရွေးစရာသုံးခု:
Gemini (အလကား) (default): aistudio.google.com
    → Get API key → key ထည့်ရုံ (native Gemini API သုံးထားပါတယ်)
OpenAI: gpt-4o-mini အကြံပြု (ပိုက်ဆံပေး)
Custom: OpenAI-compatible API တစ်ခုခု
API limit (429) တက်ရင်: ပြီးပြီးသားတွေ auto-save ဖြစ်နေတာမို့
  ၁-၂ မိနစ်စောင့်ပြီး ▶️ ဆက်လုပ် နှိပ်ပါ — ရပ်တဲ့နေရာကနေ ဆက်သွားပါတယ်။
  Request ကြားစောင့်ချိန်ကို များများထားရင် ပိုအဆင်ပြေပါတယ်
Edge TTS အသံထွက်တာ internet လိုပါတယ်
Group ရဲ့ ခန့်မှန်းအမြန်နှုန်း 1.6x ကျော်ရင် app က သတိပေးပါမယ် —
  group အရေအတွက် တိုးလိုက်ပါ
Sync: အသံကို စာတစ်ကြောင်းချင်းစီ သူ့ရဲ့ timestamp နေရာမှာ တိတိကျကျ
  ချထားတာမို့ ထပ်နေတာမရှိပါဘူး ✅။ စာကြောင်းတစ်ကြောင်း အရမ်းရှည်နေရင်
  (1) 2.0x ထိ မြန်ပေးမယ် (2) နောက်စာကြောင်းမစခင်က ရပ်နားချိန်ကို ယူသုံးမယ်
  (3) အဲဒါမှမလောက်ရင် အသံနှုန်းမြန်တာနဲ့ အလိုအလျောက်ပြန်ထုတ်ပေးမယ်
  (4) နောက်ဆုံးမှ မရမှ ဖြတ်တောက်ပြီး ဘယ်စာကြောင်းလဲဆိုတာ သတိပေးမယ် —
  အဲဒီစာကြောင်းတွေကို တိုအောင်ပြင်တာ အကောင်းဆုံးပါ
🛠️ Local မှာ run ချင်ရင်
pip install -r requirements.txt
streamlit run app.py
