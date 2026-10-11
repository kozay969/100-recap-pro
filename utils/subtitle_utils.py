import re

def parse_srt_time(time_str):
    """SRT အချိန်ကို စက္ကန့်သို့ ပြောင်းပေးသည်။"""
    hours, minutes, seconds_milliseconds = time_str.split(':')
    seconds, milliseconds = seconds_milliseconds.split(',')
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    return total_seconds

def validate_subtitle_timing(transcript_segments):
    """စာတန်းထိုး အပိုင်းအစများ၏ ကြာချိန်ကို စစ်ဆေးသည်။"""
    issues = []
    for i, seg in enumerate(transcript_segments):
        start = seg.get('start', 0) / 1000.0
        end = seg.get('end', 0) / 1000.0
        duration = end - start
        
        # အကယ်၍ စာသားရှည်ပြီး အချိန်တိုနေပါက သတိပေးရန်
        text = seg.get('text', '')
        if len(text) > 50 and duration < 2.0:
            issues.append(f"Index {i}: အချိန်တိုလွန်းသော်လည်း စာသားများပြားနေပါသည် (Duration: {duration:.2f}s)")
            
    return issues
      
