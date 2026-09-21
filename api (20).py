from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import soundfile as sf
import numpy as np
from scipy.signal import resample
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import json
import re
from difflib import SequenceMatcher
import subprocess
import threading
import time
from datetime import datetime, timezone, timedelta
import os
import urllib.request
from pydantic import BaseModel
from typing import Optional
import schedule as schedule_lib

app = FastAPI()
# Needed for browser-based admin tools (e.g. the mobile seed tool) to call
# this API from a different origin — FastAPI sends no CORS headers by
# default, so without this every fetch() from a page not served by this
# same Space would be silently blocked by the browser. This is a read/write
# admin API with no auth on /seed or /admin either way, so opening it to
# cross-origin calls doesn't change what's already reachable directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
print("Loading model...")
hf_token   = os.environ.get("HF_TOKEN", "")
processor  = WhisperProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran", token=hf_token or None)
model      = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-base-ar-quran", token=hf_token or None)
model.eval()
print("Model ready.")

# Makkah stream: cdn-globecast/akamaized went stale (frozen manifest, same
# audio served every fetch) starting ~30 Jul 2026 — confirmed dead by the
# identical-chunk-every-call symptom across 2+ days of every Makkah session.
# Swapped to the m.live.net.sa mirror (confirmed working 1 Aug 2026). If this
# one also goes stale, the other known mirror is .../live/quran/playlist.m3u8
# (same host/port) — swap the path only.
MAKKAH_STREAM  = "http://m.live.net.sa:1935/live/quran/gmswf.m3u8"
MADINAH_STREAM = "https://cdn-globecast.akamaized.net/live/eds/saudi_sunnah/hls_roku/index.m3u8"
QURAN_PATH  = "/tmp/quran.json"
TOKENS_PATH = "/tmp/push_tokens.json"
HF_DATASET  = "ittiba/ittiba-history"

CONFIDENCE_THRESHOLD = 0.44
FATIHAH_THRESHOLD    = 0.48  # balanced — low enough to catch garbled Fatihah, high enough to block false positives

OWNER_TOKEN = os.environ.get("OWNER_TOKEN", None)

from huggingface_hub import HfApi as _HfApi
from io import BytesIO

_hf_api = _HfApi(token=hf_token or None)

def hf_put(filename, data):
    try:
        content = json.dumps(data, ensure_ascii=False, indent=2).encode()
        _hf_api.upload_file(
            path_or_fileobj=BytesIO(content),
            path_in_repo=filename,
            repo_id=HF_DATASET,
            repo_type="dataset",
            commit_message=f"Update {filename}",
        )
        n = len(data) if isinstance(data, list) else "?"
        print(f"✓ HF saved {filename} ({n} sessions)")
    except Exception as e:
        print(f"HF put {filename} error: {e}")

def hf_get(filename):
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_DATASET,
            filename=filename,
            repo_type="dataset",
            token=hf_token,
            force_download=True,
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"HF get {filename} error: {e}")
        return []

_sessions_lock = threading.Lock()

def load_sessions():
    data = hf_get("frozen.json")
    print(f"Loaded: {len(data)} sessions from frozen.json")
    return data

sessions: list = load_sessions()

def save_sessions():
    hf_put("frozen.json", sessions.copy())

def upsert_session(session: dict):
    global sessions
    with _sessions_lock:
        sid = session["id"]
        idx = next((i for i, s in enumerate(sessions) if s.get("id") == sid), None)
        if idx is not None:
            sessions[idx] = session
            print(f" 🔄 Updated session {sid}")
        else:
            sessions.append(session)
            print(f" ✅ New session {sid}")
        sessions.sort(key=lambda x: (x.get("date",""), x.get("mosque",""), x.get("prayer","")))
        hf_put("frozen.json", sessions.copy())

def get_all_sessions():
    with _sessions_lock:
        return list(sessions)

_tokens_cache: list = []
_tokens_loaded = False

def load_tokens() -> list:
    global _tokens_cache, _tokens_loaded
    if _tokens_loaded: return _tokens_cache
    if os.path.exists(TOKENS_PATH):
        try: _tokens_cache = json.load(open(TOKENS_PATH)); _tokens_loaded = True; return _tokens_cache
        except: pass
    try:
        r = requests.get(f"https://huggingface.co/datasets/{HF_DATASET}/raw/main/tokens.json",
                         headers={"Authorization": f"Bearer {hf_token}"}, timeout=8)
        if r.status_code == 200:
            _tokens_cache = r.json()
            json.dump(_tokens_cache, open(TOKENS_PATH, "w"))
            print(f"✓ Loaded {len(_tokens_cache)} push tokens from HF")
    except Exception as e: print(f"Token load error: {e}")
    _tokens_loaded = True
    return _tokens_cache

def save_tokens(tokens: list):
    global _tokens_cache
    _tokens_cache = tokens
    json.dump(tokens, open(TOKENS_PATH, "w"))
    threading.Thread(target=lambda: _hf_save_tokens(tokens.copy()), daemon=True).start()

def _hf_save_tokens(tokens):
    hf_put("tokens.json", tokens)

_last_pushed = {"makkah": None, "madinah": None}

def _session_sig(s):
    r1 = s.get("rakah_1") or {}; r2 = s.get("rakah_2") or {}
    return (f"{s.get('mosque')}|{s.get('prayer')}|{s.get('date')}"
            f"|{r1.get('surah',0)}-{r1.get('ayah_start',0)}-{r1.get('ayah_end',0)}"
            f"|{r2.get('surah',0)}-{r2.get('ayah_start',0)}-{r2.get('ayah_end',0)}")

def push_all(session):
    tokens = load_tokens()
    if not tokens: return
    mosque = session.get("mosque","")
    sig = _session_sig(session)
    if _last_pushed.get(mosque) == sig:
        print(f" ⏭️ Push skipped — no change ({mosque})"); return
    _last_pushed[mosque] = sig
    mosque_name = "Makkah" if mosque == "makkah" else "Madinah"
    prayer = session.get("prayer",""); imam = session.get("imam","")
    r1 = session.get("rakah_1") or {}; r2 = session.get("rakah_2") or {}
    r1_s = f"{r1.get('surah_name','')} {r1.get('ayah_start','')}–{r1.get('ayah_end','')}" if r1 else "—"
    r2_s = f"{r2.get('surah_name','')} {r2.get('ayah_start','')}–{r2.get('ayah_end','')}" if r2 else ""
    title = f"{mosque_name} · {prayer}"
    body  = r1_s + (f"  •  {r2_s}" if r2_s else "")
    if imam: body += f"  ·  {imam}"
    try:
        requests.post("https://exp.host/--/api/v2/push/send",
            json=[{"to": t, "title": title, "body": body, "sound": "default",
                   "data": {"type": "session", "mosque": mosque, "prayer": prayer,
                            "session_id": session.get("id")}} for t in tokens], timeout=10)
        print(f"Sent ({len(tokens)} users): {title} - {body}")
    except Exception as e: print(f"Push error: {e}")

def push_owner(title, body, data=None):
    tokens = load_tokens()
    if not tokens: return
    target = OWNER_TOKEN if OWNER_TOKEN and OWNER_TOKEN in tokens else tokens[0]
    try:
        payload = {"to": target, "title": title, "body": body, "sound": "default"}
        if data: payload["data"] = data
        requests.post("https://exp.host/--/api/v2/push/send", json=[payload], timeout=10)
        print(f"Sent (owner): {title}")
    except Exception as e: print(f"Push owner error: {e}")

JUMUAH_CONFIG = {
    "week":    "May 29, 2026",
    "hijri":   "1 Dhu al-Hijjah 1447",
    "makkah": {
        "imam":  "Maher Al-Muaiqly",
        "links": {
            "arabic": "", "english": "", "urdu": "", "french": "",
            "indonesian": "", "turkish": "", "malay": "", "russian": "",
            "chinese": "", "hausa": "", "farsi": "", "hindi": "", "bengali": "",
        }
    },
    "madinah": {
        "imam":  "Abdullah Buayjaan",
        "links": {
            "arabic": "", "english": "", "urdu": "", "french": "",
            "indonesian": "", "turkish": "", "malay": "", "russian": "",
            "chinese": "", "hausa": "", "farsi": "", "hindi": "", "bengali": "",
        }
    },
    "fallback": "https://www.youtube.com/@tubesermon/streams",
}

IMAM_SCHEDULE = {
    ("makkah",  "Fajr"):    "Badr Al-Turki",
    ("makkah",  "Maghrib"): "Bandar Baleela",
    ("makkah",  "Isha"):    "Yasser Dossary",
    ("makkah",  "Jumuah"):  "Maher Al-Muaiqly",
    ("madinah", "Fajr"):    "Ahmed Al-Hudhayfi",
    ("madinah", "Maghrib"): "Salah Al-Budair",
    ("madinah", "Isha"):    "Abdul Mohsen Al-Qasim",
    ("madinah", "Jumuah"):  "Abdullah Buayjaan",
}

# ── Persistence for imam schedule + Jumuah config ─────────────────────────────
def _imam_to_disk(sched):
    out = {"makkah": {}, "madinah": {}}
    for (mosque, prayer), name in sched.items():
        out.setdefault(mosque, {})[prayer] = name
    return out

def _imam_from_disk(d):
    sched = {}
    for mosque, prayers in (d or {}).items():
        for prayer, name in (prayers or {}).items():
            sched[(mosque, prayer)] = name
    return sched

def save_imam_schedule():
    hf_put("imam_schedule.json", _imam_to_disk(IMAM_SCHEDULE))

def save_jumuah_config():
    hf_put("jumuah_config.json", JUMUAH_CONFIG)

try:
    _saved_imam = hf_get("imam_schedule.json")
    if _saved_imam:
        IMAM_SCHEDULE.update(_imam_from_disk(_saved_imam))
        print(f"✓ Loaded persisted imam schedule from HF")
except Exception as e:
    print(f"Imam schedule load skipped: {e}")

try:
    _saved_jumuah = hf_get("jumuah_config.json")
    if isinstance(_saved_jumuah, dict) and _saved_jumuah:
        JUMUAH_CONFIG.update(_saved_jumuah)
        print(f"✓ Loaded persisted Jumuah config from HF")
except Exception as e:
    print(f"Jumuah config load skipped: {e}")

def get_imam(mosque, prayer): return IMAM_SCHEDULE.get((mosque, prayer))

def quran_url(surah, a1, a2):
    return f"https://quran.com/{surah}/{a1}" if a1==a2 else f"https://quran.com/{surah}/{a1}-{a2}"

RECORDING_START = {
    "makkah":  {"Fajr": 15, "Maghrib": 6, "Isha": 8, "Jumuah": 16},
    "madinah": {"Fajr": 16, "Maghrib": 9, "Isha": 8, "Jumuah": 16},
}
RECORD_DURATION = {"Fajr": 1200, "Maghrib": 900, "Isha": 1200, "Jumuah": 1200}

PRAYER_TIMES_CACHE = {}
makkah_scheduler  = schedule_lib.Scheduler()
madinah_scheduler = schedule_lib.Scheduler()
mosque_locks = {"makkah": threading.RLock(), "madinah": threading.RLock()}

latest = {
    "makkah":  {"prayer":None,"imam":None,"rakah_1":None,"rakah_2":None,"timestamp":None},
    "madinah": {"prayer":None,"imam":None,"rakah_1":None,"rakah_2":None,"timestamp":None},
}

ARABIC_DAYS   = {"Monday":"الاثنين","Tuesday":"الثلاثاء","Wednesday":"الأربعاء","Thursday":"الخميس","Friday":"الجمعة","Saturday":"السبت","Sunday":"الأحد"}
ARABIC_MONTHS = {"January":"يناير","February":"فبراير","March":"مارس","April":"أبريل","May":"مايو","June":"يونيو","July":"يوليو","August":"أغسطس","September":"سبتمبر","October":"أكتوبر","November":"نوفمبر","December":"ديسمبر"}

def get_saudi_now(): return datetime.now(timezone(timedelta(hours=3)))
def is_friday_saudi(): return get_saudi_now().weekday() == 4

def get_arabic_date(dt=None):
    if dt is None: dt = get_saudi_now()
    if isinstance(dt, str):
        try: dt = datetime.fromisoformat(dt).replace(tzinfo=timezone(timedelta(hours=3)))
        except: dt = get_saudi_now()
    d, m = dt.strftime("%A"), dt.strftime("%B")
    return {"gregorian": f"{d}, {dt.day} {m} {dt.year}",
            "arabic":    f"{ARABIC_DAYS.get(d,d)}، {dt.day} {ARABIC_MONTHS.get(m,m)} {dt.year}",
            "day": d, "arabic_day": ARABIC_DAYS.get(d, d)}

_HIJRI_MONTHS_EN = ["Muharram","Safar","Rabi al-Awwal","Rabi al-Thani","Jumada al-Awwal","Jumada al-Thani",
                     "Rajab","Shaban","Ramadan","Shawwal","Dhu al-Qadah","Dhu al-Hijjah"]
_HIJRI_MONTHS_AR = ["محرم","صفر","ربيع الأول","ربيع الآخر","جمادى الأولى","جمادى الآخرة",
                     "رجب","شعبان","رمضان","شوال","ذو القعدة","ذو الحجة"]

def _hijri_tabular_fallback(dt):
    # Dependency-free local Hijri conversion (standard Julian-Day-Number-based
    # tabular/civil Islamic calendar) — used only when the Aladhan API call
    # fails, so the Hijri date is NEVER simply missing regardless of what the
    # external API is doing. Can differ from the Umm al-Qura date the API
    # normally returns by up to ~1-2 days — the same margin Aladhan itself
    # documents as inherent to any Gregorian↔Hijri conversion.
    y, m, d = dt.year, dt.month, dt.day
    a = (14 - m) // 12
    y2 = y + 4800 - a
    m2 = m + 12*a - 3
    jdn = d + (153*m2 + 2)//5 + 365*y2 + y2//4 - y2//100 + y2//400 - 32045
    jdn = jdn - 1948440 + 10632
    n = (jdn - 1) // 10631
    jdn = jdn - 10631*n + 354
    j = ((10985 - jdn)//5316)*((50*jdn)//17719) + (jdn//5670)*((43*jdn)//15238)
    jdn = jdn - ((30 - j)//15)*((17719*j)//50) - (j//16)*((15238*j)//43) + 29
    im = (24*jdn)//709
    idd = jdn - (709*im)//24
    iy = 30*n + j - 30
    mi = max(0, min(11, im-1))
    return {"day": str(idd), "month": _HIJRI_MONTHS_EN[mi], "month_ar": _HIJRI_MONTHS_AR[mi], "year": str(iy),
            "formatted": f"{idd} {_HIJRI_MONTHS_EN[mi]} {iy}", "formatted_ar": f"{idd} {_HIJRI_MONTHS_AR[mi]} {iy}"}

_hijri_cache = {"date": None, "data": None}
def get_hijri_cached():
    # Bug fixed (round 1): this used to stamp _hijri_cache["date"] = today
    # even when the request failed (timeout/non-200/bad JSON), because that
    # line sat outside the success branch. One transient aladhan.com hiccup
    # would then poison the cache for the entire rest of the day.
    # Bug fixed (round 2): even with that fixed, if the API is *persistently*
    # unreachable from this environment (not just a one-off blip), every call
    # still returns None forever with nothing to show for it. So: on any
    # failure, compute a local fallback (see _hijri_tabular_fallback above)
    # and return that instead of None, while still leaving the date unstamped
    # so the API gets retried — and preferred — on the next call once it's
    # back. The Hijri field should now never be empty, full stop.
    today_dt = get_saudi_now()
    today = today_dt.strftime("%Y-%m-%d")
    if _hijri_cache["date"] != today:
        got_api = False
        try:
            r = requests.get(f"https://api.aladhan.com/v1/gToH/{today_dt.strftime('%d-%m-%Y')}", timeout=5)
            if r.status_code == 200:
                d = r.json()["data"]["hijri"]
                _hijri_cache["data"] = {"day":d["day"],"month":d["month"]["en"],"month_ar":d["month"]["ar"],"year":d["year"],
                    "formatted":f"{d['day']} {d['month']['en']} {d['year']}","formatted_ar":f"{d['day']} {d['month']['ar']} {d['year']}"}
                _hijri_cache["date"] = today
                got_api = True
            else:
                print(f" ⚠️ Hijri date fetch failed: HTTP {r.status_code} — using local fallback")
        except Exception as e:
            print(f" ⚠️ Hijri date fetch error: {e} — using local fallback")
        if not got_api:
            _hijri_cache["data"] = _hijri_tabular_fallback(today_dt)
    return _hijri_cache["data"]

def add_minutes(t, mins):
    h,m = map(int,t.split(":")); total = h*60+m+mins
    return f"{total//60%24:02d}:{total%60:02d}"

def saudi_to_utc(t):
    h,m = map(int,t.split(":")); total = h*60+m-180
    if total<0: total+=1440
    return f"{total//60%24:02d}:{total%60:02d}"

def get_prayer_times(city):
    key = city+"_"+get_saudi_now().strftime("%Y-%m-%d")
    if key in PRAYER_TIMES_CACHE: return PRAYER_TIMES_CACHE[key]
    try:
        r = requests.get(f"https://api.aladhan.com/v1/timingsByCity?city={city}&country=Saudi Arabia&method=4", timeout=10)
        d = r.json()
        if "data" in d: PRAYER_TIMES_CACHE[key] = d["data"]["timings"]; return PRAYER_TIMES_CACHE[key]
    except Exception as e: print(f"Prayer times error {city}: {e}")
    return {"Fajr":"04:42","Dhuhr":"12:22","Asr":"15:44","Maghrib":"18:47","Isha":"20:17"}

def get_next_prayer(times):
    now = get_saudi_now().strftime("%H:%M")
    for p in ["Fajr","Dhuhr","Asr","Maghrib","Isha"]:
        if times.get(p,"") > now: return {"name":p,"time":times[p]}
    return {"name":"Fajr","time":times.get("Fajr","")}

def session_id(mosque, prayer, date): return f"{date}_{mosque}_{prayer}"

def populate_latest():
    all_s = get_all_sessions()
    PRAYER_ORDER = {"Fajr":0,"Jumuah":1,"Dhuhr":1,"Asr":2,"Maghrib":3,"Isha":4}
    for mosque in ["makkah","madinah"]:
        ms = [s for s in all_s if s.get("mosque")==mosque]
        if not ms: continue
        recent = max(ms, key=lambda s:(s.get("date",""), PRAYER_ORDER.get(s.get("prayer",""),0)))
        imam = recent.get("imam")
        def rw(r): return {**r,"imam":imam} if r else None
        latest[mosque] = {"prayer":recent.get("prayer"),"imam":imam,
                          "rakah_1":rw(recent.get("rakah_1")),"rakah_2":rw(recent.get("rakah_2")),
                          "timestamp":(recent.get("rakah_1") or {}).get("timestamp"),"seeded":True}
        print(f"📋 Loaded latest {mosque}: {recent.get('prayer')} on {recent.get('date')} ({imam})")

populate_latest()

# ── Audio ─────────────────────────────────────────────────────────────────────
TARGET_SAMPLES = 480000
_transcribe_lock = threading.Lock()  # Whisper is not thread-safe — one inference at a time

def sample_audio(mosque, secs=20):
    stream = MAKKAH_STREAM if mosque=="makkah" else MADINAH_STREAM
    path   = f"/tmp/live_{mosque}.wav"
    section = f"*00:00:00-00:{secs//60:02d}:{secs%60:02d}"
    fetch_timeout = max(60, secs * 3)
    try:
        subprocess.run(["yt-dlp","-x","--audio-format","wav","--postprocessor-args","-ar 16000 -ac 1",
                        "-o",path,stream,"--download-sections",section,"--force-overwrites"],
                       capture_output=True, text=True, timeout=fetch_timeout)
    except subprocess.TimeoutExpired:
        print(f" ⏱️ sample_audio timeout ({fetch_timeout}s) — {mosque} stream stalled, skipping chunk")
        return None
    except Exception as e:
        print(f" ⚠️ sample_audio error ({mosque}): {e}")
        return None
    if not os.path.exists(path): return None
    # Everything from here on (read + shape/format fixes) is one failure unit:
    # a truncated/corrupt WAV from a partial fetch can blow up in resample()/
    # astype() just as easily as in sf.read(), and those calls used to sit
    # outside this try — an exception there would propagate out of sample_audio
    # uncaught, up through record_prayer's while loop (which has no guard of
    # its own), silently killing that mosque's whole recording thread with no
    # traceback. (Suspected cause of the 3 Sep Madinah Fajr session: audio
    # started fine, then all logging for that thread just stopped mid-window,
    # no error, no completion, nothing.) Catching the whole block fixes that
    # failure mode regardless of which exact line trips it.
    try:
        audio, sr = sf.read(path)
        if len(audio.shape)>1: audio=audio.mean(axis=1)
        if sr!=16000: audio=resample(audio,int(len(audio)*16000/sr))
        return audio.astype(np.float32)
    except Exception as e:
        print(f" ⚠️ sample_audio read/convert error ({mosque}): {e}")
        return None

def transcribe(chunk):
    import gc
    try:
        if len(chunk)<16000*0.5: return ""
        if len(chunk)<TARGET_SAMPLES: chunk=np.pad(chunk,(0,TARGET_SAMPLES-len(chunk)))
        else: chunk=chunk[:TARGET_SAMPLES]
        got = _transcribe_lock.acquire(timeout=90)
        if not got:
            print(" ⏱️ transcribe lock timeout — skipping chunk")
            return ""
        try:
            with torch.inference_mode():
                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt")
                feat = inputs["input_features"]
                if feat.shape[-1]!=3000:
                    if feat.shape[-1]>3000: feat=feat[...,:3000]
                    else: feat=torch.cat([feat,torch.zeros(*feat.shape[:-1],3000-feat.shape[-1])],dim=-1)
                    inputs["input_features"]=feat
                gen = model.generate(**inputs, max_new_tokens=256)
            result = processor.batch_decode(gen, skip_special_tokens=True)[0]
        finally:
            _transcribe_lock.release()
        gc.collect()
        return result
    except Exception as e: print(f" Transcription error: {e}"); gc.collect(); return ""

import unicodedata as _ud
def strip_diacritics(s):
    if not s: return ""
    s = _ud.normalize("NFKD", s)
    s = "".join(c for c in s if not _ud.combining(c))
    s = re.sub(r"[\u0600-\u0605\u0610-\u061A\u06D6-\u06ED\u08D3-\u08FF\u0640\u06DD۞]", "", s)
    s = (s.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ٱ","ا")
           .replace("ة","ه").replace("ى","ي").replace("ؤ","و").replace("ئ","ي"))
    return re.sub(r"\s+", " ", s).strip()

def sim(text, phrases):
    if not text: return 0
    t = strip_diacritics(text)
    return max(SequenceMatcher(None,t,strip_diacritics(p)).ratio() for p in phrases)

FATIHAH_PHRASES = [
    "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ الرَّحْمَنِ الرَّحِيمِ",
    "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
    "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
    "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
    "غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
    "مَالِكِ يَوْمِ الدِّينِ إِيَّاكَ نَعْبُدُ",
    "الرَّحْمَنِ الرَّحِيمِ مَالِكِ يَوْمِ الدِّينِ",
    "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ",
    "الحمد لله رب العالمين",
    "إياك نعبد وإياك نستعين",
    "اهدنا الصراط المستقيم",
]
TAKBEER_PHRASES = ["اللَّهُ أَكْبَرُ اللَّهُ أَكْبَرُ اللَّهُ أَكْبَرُ",
    "سُبْحَانَ رَبِّيَ الْعَظِيمِ وَبِحَمْدِهِ","سُبْحَانَ رَبِّيَ الْأَعْلَى وَبِحَمْدِهِ",
    "سَمِعَ اللَّهُ لِمَنْ حَمِدَهُ رَبَّنَا وَلَكَ الْحَمْدُ"]
SALAM_PHRASES  = ["السَّلَامُ عَلَيْكُمْ وَرَحْمَتُ اللَّهِ","السلام عليكم ورحمة الله"]

_TAKBEER_TOKENS = ["اكبر","عكبا","عكب","اكبا","اكثر","اكمر","عكم","اكبار","حمده","الاواب","وكبر","وكبا","عتوي"]
_ALLAH_WORDS = {"الله","لله","والله","اللهم"}
def is_takbeer_fragment(text):
    if not text: return False
    words = strip_diacritics(text).split()
    if not words: return False
    if not any(any(tok in w for tok in _TAKBEER_TOKENS) for w in words):
        return False
    hits = sum(1 for w in words if any(tok in w for tok in _TAKBEER_TOKENS) or w in _ALLAH_WORDS)
    return len(words) <= 6 and hits >= max(1, len(words) - 1)

_DUA_MARKERS = [
    "اغفر لنا","ارحم موتانا","ارحمنا برحمتك","اللهم اجعلنا","اللهم اجعلن",
    "بفضلك وكرمك","يا اكرم الاكرمين","ثواب اعمالنا","تجاوز عن ذنوب",
    "نفوسا راضيه","الوسيله والفضيله","حتى ترضى","يوفي بعهدك","عصمنا",
    "وفرجه","امدن الوسيله","وابعثه","وارزقنا","قلوبا خاشعه",
    "يا ارحم الراحمين","اجعل ثواب","صل على محمد","صل وسلم","بحبك صافيه",
    "نسالك علما نافعا","علما نافعا","وعافنا","ولا تزغ قلوبنا",
    "بعد اذ هديتنا","موتانا","وامدن",
]
def is_dua_fragment(text):
    if not text: return False
    n = strip_diacritics(text)
    return sum(1 for m in _DUA_MARKERS if m in n) >= 2

def classify(text):
    if not text: return "empty"
    if sim(text,SALAM_PHRASES)   >= 0.6:  return "salam"
    if sim(text,FATIHAH_PHRASES) >= FATIHAH_THRESHOLD: return "fatihah"
    if sim(text,TAKBEER_PHRASES) >= 0.65: return "takbeer"
    if is_takbeer_fragment(text): return "takbeer"
    if is_dua_fragment(text): return "dua"
    return "quran"

# ── Recording ─────────────────────────────────────────────────────────────────
def record_prayer(mosque, prayer_name):
    total_secs = RECORD_DURATION.get(prayer_name, 900)
    chunk_secs = 20; classified = []; elapsed = 0
    start_offset_min = RECORDING_START.get(mosque, {}).get(prayer_name, 0)
    first_fatihah_logged = False
    wall_deadline = time.time() + total_secs + 600  # audio length + 10 min slack
    print(f" 📼 Recording {total_secs}s...")
    while elapsed < total_secs:
        if time.time() > wall_deadline:
            print(f" ⏱️ Wall-clock deadline hit at {elapsed}s — stopping, will vote on {len(classified)} chunks")
            return classified
        # Whole-iteration guard: a single bad chunk (corrupt audio, an
        # unexpected exception anywhere in the transcribe/classify path)
        # must never be able to kill this loop and silently drop the entire
        # prayer session with zero output and zero error — log it and keep
        # going instead. (This is the fix for the 3 Sep Madinah Fajr session,
        # which recorded a few real chunks then produced no further output
        # at all — no error, no completion, nothing.)
        try:
            audio = sample_audio(mosque, secs=chunk_secs)
            if audio is None: elapsed+=chunk_secs; continue
            for i in range(0, len(audio), 16000*chunk_secs):
                chunk = audio[i:i+16000*chunk_secs]
                if len(chunk)<16000*3: continue
                text=transcribe(chunk); cls=classify(text)
                classified.append((chunk,text,cls))
                print(f" [{elapsed}s|{cls}] {text[:60]}...")
                if cls=="fatihah" and not first_fatihah_logged:
                    first_fatihah_logged = True
                    secs_after_adhan = start_offset_min*60 + elapsed
                    clock = get_saudi_now().strftime("%H:%M:%S")
                    print(f" 🎯 First Fatihah at {clock} Saudi — recitation began "
                          f"~{secs_after_adhan}s after adhan ({secs_after_adhan/60:.1f} min). "
                          f"Current offset: {start_offset_min} min.")
                if cls=="salam" and elapsed>=180:
                    print(f" ✅ Salam at {elapsed}s — done"); del audio; return classified
            del audio
        except Exception as e:
            import traceback
            print(f" 💥 chunk error in record_prayer ({mosque} {prayer_name}, elapsed={elapsed}s): {e}")
            traceback.print_exc()
        elapsed+=chunk_secs
        print(f" ⏱️ {elapsed}s / {total_secs}s...")
    return classified

# ── Rakah detection ───────────────────────────────────────────────────────────
def find_first_fatihah(classified, search_from=0):
    for i in range(search_from, len(classified)):
        if classified[i][2]!="fatihah": continue
        qa = sum(1 for _,_,lc in classified[i+1:i+9] if lc=="quran")
        if qa>=2: print(f" ✅ F1 at chunk {i} ({qa} quran after)"); return i
        print(f" ⏭️  Skip Fatihah {i} ({qa} quran after)")
    return None

def find_second_fatihah(classified, f1, prayer_name):
    MIN_GAP  = max(8, 6 if prayer_name=="Maghrib" else 10)
    MIN_QRAN = 1 if prayer_name=="Maghrib" else 2
    for i in range(f1+MIN_GAP, len(classified)):
        if classified[i][2]!="fatihah": continue
        qa = sum(1 for _,_,lc in classified[i+1:i+7] if lc=="quran")
        if qa>=MIN_QRAN: print(f" ✅ F2 at chunk {i} ({qa} quran after)"); return i
        print(f" ⏭️  Skip ruku Fatihah {i}")
    return None

def parse_rakahs(classified, prayer_name):
    f1 = find_first_fatihah(classified)
    if f1 is None:
        all_q = [(c,t) for c,t,cl in classified if cl=="quran"]
        print(f" ⚠️ No F1 — {len(all_q)} quran chunks")
        qi = [i for i,(_,_,cl) in enumerate(classified) if cl=="quran"]
        if len(qi)>4:
            for j in range(1,len(qi)):
                if qi[j]-qi[j-1]>=4:
                    bs=j; print(f" 🔍 Gap split at {j}"); break
            else: bs=None
            if bs and bs>1:
                return all_q[:bs], all_q[bs:]
        if len(all_q)>6:
            mid=int(len(all_q)*(0.6 if prayer_name in ("Isha","Fajr") else 0.5))
            return all_q[:mid], all_q[mid:]
        return all_q, []

    f2 = find_second_fatihah(classified, f1, prayer_name)
    r1 = [(c,t) for c,t,cl in classified[f1+1:f2 if f2 else len(classified)] if cl=="quran"]
    print(f" R1: {len(r1)} chunks")
    r2 = []
    if f2 is not None:
        start=f2+1
        while start<len(classified) and classified[start][2]=="takbeer": start+=1
        for i in range(start,len(classified)):
            cl=classified[i][2]
            if cl=="salam" and len(r2)>=1: break
            if cl=="quran": r2.append((classified[i][0],classified[i][1]))
    print(f" R2: {len(r2)} chunks")
    if not r1 and r2: return r2,[]
    thr=8 if prayer_name=="Maghrib" else 12
    if len(r2)<3 and len(r1)>thr:
        mid=int(len(r1)*(0.6 if prayer_name in ("Isha","Fajr") else 0.5))
        print(f" ✂️ Split R1 at {mid}"); return r1[:mid],r1[mid:]
    return r1,r2

# ── Quran data ────────────────────────────────────────────────────────────────
_QURAN_DATA = None
def get_quran():
    global _QURAN_DATA
    if _QURAN_DATA is None:
        if not os.path.exists(QURAN_PATH):
            urllib.request.urlretrieve("https://cdn.jsdelivr.net/npm/quran-json@3.1.2/dist/quran.json", QURAN_PATH)
        _QURAN_DATA = json.load(open(QURAN_PATH, encoding="utf-8"))
        for _s in _QURAN_DATA:
            for _v in _s["verses"]:
                _v["_norm"] = strip_diacritics(_v["text"])
    return _QURAN_DATA

get_quran()

SHORT_SURAH_IDS  = {112,108,110,109,111,113,114,103,107,105,106,104}
MEDIUM_SHORT_IDS = {95,94,93,91,90,89,88,87,86,85,84,83,61,62,63,64,66,67,99,100,101,102,103,104}

NOISE_PHRASES = {"وَالْمُؤْمِنِينَ","والمؤمنين","وَالْمُسْلِمِينَ","وَالْمُسْلِمُ",
                 "وَمَا يَغِيظُ","وَمَا يَغْفَرُونَ","وَمَا يَسْطَعُونَ","وَمَا يَسْرِي"}

_ctc_proc=None; _ctc_mdl=None

def _load_ctc():
    global _ctc_proc, _ctc_mdl
    if _ctc_proc: return True
    try:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        print("🔧 Loading CTC model...")
        _ctc_proc = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
        _ctc_mdl  = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-arabic")
        _ctc_mdl.eval(); print("✓ CTC model loaded"); return True
    except Exception as e: print(f"⚠️ CTC unavailable: {e}"); return False

def ctc_rerank(chunks, ranked, quran, top_k=5):
    if not _load_ctc() or not ranked or not chunks: return None
    combined = np.concatenate([a for a,_ in chunks])[:16000*30]
    if len(combined)<16000: return None
    print(f" 🔬 CTC re-ranking top {min(top_k,len(ranked))} candidates...")
    best_sid=None; best_score=float("-inf")
    with torch.inference_mode():
        inputs = _ctc_proc(combined, sampling_rate=16000, return_tensors="pt", padding=True)
        logits = _ctc_mdl(**inputs).logits
        lp = torch.nn.functional.log_softmax(logits, dim=-1)
    for sid,vi in ranked[:top_k]:
        sd = next((s for s in quran if s["id"]==sid),None)
        if not sd: continue
        ayahs=[a[0] if isinstance(a,(tuple,list)) else a for a in vi.get("ayahs",[])]
        if not ayahs: continue
        lo=min(ayahs); hi=max(ayahs)
        texts=[v["text"] for v in sd["verses"] if lo<=v["id"]<=hi]
        if not texts: continue
        toks=_ctc_proc.tokenizer.encode(strip_diacritics(" ".join(texts)),add_special_tokens=False)
        if not toks: continue
        score=sum(lp[0,:,t].max().item() for t in toks if t<lp.shape[-1])/max(len(toks),1)
        print(f"   {vi['name']}: CTC={score:.3f}")
        if score>best_score: best_score=score; best_sid=sid
    if best_sid: print(f" 🎯 CTC winner: surah {best_sid}")
    return best_sid

def min_vote_count(n, prayer):
    if prayer=="Maghrib": return 0.5
    if n<=2: return 1.0
    if n<=6: return 1.5
    return 2.0

def vote_surah(chunks, continuity_id=None, prayer_name="Isha", r1_ayah_end=0):
    if not chunks: return None
    quran=get_quran(); votes={}; total=len(chunks); mvc=min_vote_count(total,prayer_name)
    print(f" 🗳️ Voting {total} chunks (min_votes={mvc})...")
    if total>=4:
        from collections import Counter
        texts=[t.strip() for _,t in chunks]
        top_t,top_c=Counter(texts).most_common(1)[0]
        if top_c/total>0.50: print(f" ⚠️ Noise: '{top_t[:30]}' {top_c}/{total} — skip"); return None
    short_min = 0.65 if total>=5 else 0.45
    for idx,(chunk,text) in enumerate(chunks):
        cls=classify(text); print(f" [{cls}] → {text[:55]}...")
        if cls in ("takbeer","fatihah","salam","empty","dua"): continue
        if text.strip() in NOISE_PHRASES or len(text.strip().split())<=2: continue
        w=0.7 if (total>=3 and (idx==0 or idx==total-1)) else 1.3 if total>=3 else 1.0
        best=None
        ntext=strip_diacritics(text)
        for surah in quran:
            if surah["id"]==1: continue
            for ayah in surah["verses"]:
                s=SequenceMatcher(None,ntext,ayah["_norm"]).ratio()
                if best is None or s>best["confidence"]:
                    best={"surah":surah["id"],"surah_name":surah["transliteration"],"ayah":ayah["id"],"confidence":round(s,2)}
        if not best or best["confidence"]<0.3: continue
        sid=best["surah"]
        if sid in SHORT_SURAH_IDS and best["confidence"]<short_min and prayer_name!="Maghrib": continue
        if sid in MEDIUM_SHORT_IDS and total>=8 and best["confidence"]<0.62 and prayer_name!="Maghrib": continue
        if sid not in votes: votes[sid]={"count":0.0,"name":best["surah_name"],"max_conf":0,"ayahs":[]}
        is_cont=continuity_id and sid==continuity_id
        boost=2.5 if is_cont else 1.0
        votes[sid]["count"]+=w*boost
        if not (is_cont and r1_ayah_end>0 and best["ayah"]<r1_ayah_end-10):
            votes[sid]["ayahs"].append((best["ayah"], best["confidence"]))
        if best["confidence"]>votes[sid]["max_conf"]: votes[sid]["max_conf"]=best["confidence"]

    if not votes: return None
    if continuity_id and continuity_id in votes:
        cc=votes[continuity_id]["max_conf"]
        if cc>=0.45: votes={sid:v for sid,v in votes.items() if v["max_conf"]>=cc*0.7 or sid==continuity_id}
        elif cc>=0.35:
            cv=votes[continuity_id]["count"]
            votes={sid:v for sid,v in votes.items() if sid==continuity_id or v["count"]>=cv*0.8}
    votes={sid:v for sid,v in votes.items() if v["count"]>=mvc}
    if not votes: print(" ⚠️ All below min_vote_count"); return None

    ranked=sorted(votes.items(),key=lambda kv:kv[1]["count"]*0.6+kv[1]["max_conf"]*0.4,reverse=True)
    best_id=ranked[0][0]; b=votes[best_id]
    print(f" 🏆 {b['name']} ({b['count']:.1f} votes, max conf: {b['max_conf']})")

    SHORT_SURAH_MIN = 78
    if best_id >= SHORT_SURAH_MIN:
        floor_conf, floor_votes = 0.50, 2.0
    else:
        floor_conf, floor_votes = 0.60, 4.0
    if b["max_conf"] < floor_conf and b["count"] < floor_votes:
        print(f" ❌ Weak winner rejected (conf {b['max_conf']} < {floor_conf}, votes {b['count']:.1f} < {floor_votes} | surah {best_id})")
        return None

    if len(ranked) >= 2:
        top_score = ranked[0][1]["count"]*0.6 + ranked[0][1]["max_conf"]*0.4
        sec_score = ranked[1][1]["count"]*0.6 + ranked[1][1]["max_conf"]*0.4
        vote_lead = (top_score - sec_score) / max(top_score, 0.01)
        print(f" ⏭️ CTC disabled — Whisper lead {vote_lead*100:.0f}%")
    else:
        print(f" ⏭️ CTC disabled — single candidate")

    pairs = sorted(b["ayahs"], key=lambda p: p[0])

    if pairs:
        peak_conf = max(c for _, c in pairs)
        floor = max(0.4, peak_conf * 0.7)
        strong = [p for p in pairs if p[1] >= floor]
        if len(strong) >= 1:
            pairs = strong
    n = len(pairs)

    CLUSTER_WIN = {"Maghrib": 15, "Fajr": 30, "Isha": 30, "Jumuah": 40}.get(prayer_name, 30)

    if n <= 2:
        a1 = pairs[0][0]; a2 = pairs[-1][0]
    else:
        anchor = max(pairs, key=lambda p: p[1])[0]
        cluster = [p for p in pairs if abs(p[0] - anchor) <= CLUSTER_WIN]
        if len(cluster) < 2:
            cluster = pairs
        cluster.sort(key=lambda p: p[0])
        total_w = sum(c for _, c in cluster) or 1.0
        acc = 0.0; lo_i = 0
        for i, (_, c) in enumerate(cluster):
            acc += c
            if acc >= total_w * 0.10: lo_i = i; break
        acc = 0.0; hi_i = len(cluster) - 1
        for i in range(len(cluster) - 1, -1, -1):
            acc += cluster[i][1]
            if acc >= total_w * 0.10: hi_i = i; break
        a1 = cluster[lo_i][0]; a2 = cluster[max(hi_i, lo_i)][0]
    raw_lo = pairs[0][0]; raw_hi = pairs[-1][0]
    print(f" 📐 Ayah range: {a1}–{a2} (clustered from {raw_lo}–{raw_hi} raw, n={n})")

    MAX_SPAN={"Maghrib":40,"Fajr":80,"Isha":80,"Jumuah":100}
    span=a2-a1
    if span>MAX_SPAN.get(prayer_name,80):
        mid=(a1+a2)//2; hs=MAX_SPAN.get(prayer_name,80)//2
        a1=max(1,mid-hs); a2=a1+MAX_SPAN.get(prayer_name,80)
        print(f" ✂️ Capped span → {a1}–{a2}")

    sd=next((s for s in quran if s["id"]==best_id),None)
    if sd:
        ta=len(sd["verses"])
        if 1<a1<=5: print(f" 📎 Start snap → 1"); a1=1
        if ta<19 and b["max_conf"]>=0.60:
            if a1>1 or a2<ta: print(f" 📎 Short surah snap → 1–{ta}"); a1=1; a2=ta
        elif ta<19: print(f" ⚠️ Short surah snap skipped (conf {b['max_conf']} < 0.60)")

    return {"surah":best_id,"surah_name":b["name"],"ayah":a1,"ayah_end":a2,
            "confidence":round(min(0.5+b["count"]*0.08,0.95),2)}

# ── Save detected session ─────────────────────────────────────────────────────
def save_detected(mosque, prayer_name, r1_result, r2_result):
    now=get_saudi_now(); date=now.strftime("%Y-%m-%d")
    di=get_arabic_date(now); imam=get_imam(mosque,prayer_name)
    sid=session_id(mosque,prayer_name,date)

    def rakah_obj(r):
        if not r: return None
        return {"surah":r["surah"],"surah_name":r["surah_name"],
                "ayah_start":r["ayah"],"ayah_end":r["ayah_end"],
                "quran_url":quran_url(r["surah"],r["ayah"],r["ayah_end"]),
                "confidence":r["confidence"],"timestamp":datetime.now().isoformat()}

    session = {
        "id":sid,"mosque":mosque,"prayer":prayer_name,"date":date,
        "gregorian_date":di["gregorian"],"arabic_date":di["arabic"],
        "day":di["day"],"arabic_day":di["arabic_day"],"hijri":get_hijri_cached(),
        "imam":imam,"rakah_1":rakah_obj(r1_result),"rakah_2":rakah_obj(r2_result),
        "auto_detected":True,"confirmed":False,
    }

    upsert_session(session)
    mosque_name="Makkah" if mosque=="makkah" else "Madinah"
    def _wi(r): return {**r,"imam":imam} if r else None
    latest[mosque]={"prayer":prayer_name,"imam":imam,
                    "rakah_1":_wi(session["rakah_1"]),
                    "rakah_2":_wi(session["rakah_2"]),
                    "timestamp":datetime.now().isoformat(),
                    "seeded":False}

    push_all(session)

    r1_s=f"{r1_result['surah_name']} {r1_result['ayah']}–{r1_result['ayah_end']}" if r1_result else "—"
    r2_s=f"{r2_result['surah_name']} {r2_result['ayah']}–{r2_result['ayah_end']}" if r2_result else "—"
    body=f"R1: {r1_s}  R2: {r2_s}" + (f" · {imam}" if imam else "")
    push_owner(f"📋 {mosque_name} · {prayer_name} — correct?", body,
               data={"type":"confirm_draft","session_id":sid,"mosque":mosque,"prayer":prayer_name})

# ── Main listener ─────────────────────────────────────────────────────────────
def smart_listen(mosque, prayer_name):
    if prayer_name=="Jumuah" and not is_friday_saudi():
        print(f" ⏭️ Jumuah skipped ({get_saudi_now().strftime('%A')})"); return
    lock=mosque_locks[mosque]
    if not lock.acquire(blocking=True,timeout=60): print(f" ⚠️ {mosque} lock timeout"); return
    try:
        print(f"\n🕌 {mosque.upper()} {prayer_name} — {get_saudi_now().strftime('%H:%M %A')} Saudi")
        try:
            classified=record_prayer(mosque,prayer_name)
            r1,r2=parse_rakahs(classified,prayer_name)
        except Exception as e:
            import traceback
            print(f" 💥 ERROR in record/parse for {mosque} {prayer_name}: {e}")
            traceback.print_exc()
            # Surface this to the owner instead of it only living in logs —
            # a session that silently vanishes (like 3 Sep Madinah Fajr) is
            # otherwise invisible until someone happens to check.
            try:
                push_owner(f"⚠️ {mosque.title()} {prayer_name} failed",
                           f"Recording/parsing error: {e}")
            except Exception:
                pass
            return

        print(f"\n === Rakah 1 ({len(r1)} chunks) ===")
        r1_result=vote_surah(r1,prayer_name=prayer_name)
        r1_min=0.1 if prayer_name=="Maghrib" else CONFIDENCE_THRESHOLD
        if r1_result and r1_result["confidence"]>=r1_min: print(f" ✅ R1: {r1_result['surah_name']} {r1_result['ayah']}–{r1_result['ayah_end']}")
        else: r1_result=None; print(" ❌ R1: low confidence")

        r1_reliable=r1_result is not None and r1_result["confidence"]>=0.50

        print(f"\n === Rakah 2 ({len(r2)} chunks) ===")
        r1_surah=r1_result["surah"] if r1_result else None
        if r1_surah: print(f" 💡 Continuity: surah {r1_surah}")
        r2_result=vote_surah(r2,continuity_id=(r1_surah if r1_reliable else None),
                             prayer_name=prayer_name,r1_ayah_end=(r1_result["ayah_end"] if r1_result else 0))
        r2_min=0.1 if prayer_name=="Maghrib" else CONFIDENCE_THRESHOLD
        if r2_result and r2_result["confidence"]>=r2_min: print(f" ✅ R2: {r2_result['surah_name']} {r2_result['ayah']}–{r2_result['ayah_end']}")
        else: r2_result=None; print(" ❌ R2: low confidence")

        if r1_result and r2_result and r1_result["surah"]==r2_result["surah"]:
            r1e=r1_result["ayah_end"]; r2s=r2_result["ayah"]
            if r2s>r1e+1 and r2s-r1e-1<=3:
                print(f" 🔗 Gap bridge: R1 end {r1e}→{r2s-1}")
                r1_result=dict(r1_result); r1_result["ayah_end"]=r2s-1
            elif r2s<=r1e:
                print(f" 🔗 Overlap fix: R2 start {r2s}→{r1e+1}")
                r2_result=dict(r2_result); r2_result["ayah"]=r1e+1
                if r2_result["ayah_end"] < r2_result["ayah"]:
                    r2_result["ayah_end"] = r2_result["ayah"]

        if r1_result or r2_result:
            save_detected(mosque,prayer_name,r1_result,r2_result)
    finally:
        lock.release()

# ── Scheduler ─────────────────────────────────────────────────────────────────
def run_scheduler(mosque, city, sched):
    while True:
        sched.clear()
        times=get_prayer_times(city); start=RECORDING_START[mosque]
        for prayer,key in [("Fajr","Fajr"),("Maghrib","Maghrib"),("Isha","Isha")]:
            t=saudi_to_utc(add_minutes(times[prayer],start[prayer]))
            sched.every().day.at(t).do(lambda m=mosque,p=prayer: threading.Thread(target=smart_listen,args=(m,p)).start())
        t_j=saudi_to_utc(add_minutes(times["Dhuhr"],start["Jumuah"]))
        sched.every().day.at(t_j).do(lambda m=mosque: threading.Thread(target=smart_listen,args=(m,"Jumuah")).start())
        di=get_arabic_date()
        print(f"{mosque.upper()} — {di['gregorian']}")
        print(f"  Fajr@{add_minutes(times['Fajr'],start['Fajr'])}  "
              f"Maghrib@{add_minutes(times['Maghrib'],start['Maghrib'])}  "
              f"Isha@{add_minutes(times['Isha'],start['Isha'])}  "
              f"Jumuah@{add_minutes(times['Dhuhr'],start['Jumuah'])} (Fri only) Saudi")
        midnight=time.time()+86400
        while time.time()<midnight: sched.run_pending(); time.sleep(30)

threading.Thread(target=run_scheduler,args=("makkah","Makkah",makkah_scheduler),daemon=True).start()
threading.Thread(target=run_scheduler,args=("madinah","Madinah",madinah_scheduler),daemon=True).start()

# ── API ───────────────────────────────────────────────────────────────────────
class TokenReq(BaseModel): token: str

class RakahIn(BaseModel):
    surah: int; surah_name: str; ayah_start: int; ayah_end: int
    quran_url: Optional[str]=None; confidence: Optional[float]=0.95; timestamp: Optional[str]=None

class SessionIn(BaseModel):
    mosque: str; prayer: str; date: str; imam: Optional[str]=None
    rakah_1: Optional[RakahIn]=None; rakah_2: Optional[RakahIn]=None
    rakah_2b: Optional[RakahIn]=None

@app.post("/register")
def register(req: TokenReq):
    if req.token=="test": return {"status":"registered"}
    tokens=load_tokens()
    if req.token not in tokens:
        tokens.append(req.token); save_tokens(tokens)
        print(f"Registered: {req.token} (total: {len(tokens)})")
    return {"status":"registered","total_tokens":len(tokens)}

@app.get("/tokens")
def get_tokens():
    tokens=load_tokens(); return {"count":len(tokens),"tokens":[t[:20]+"..." for t in tokens]}

@app.post("/seed/session")
def seed_session(req: SessionIn):
    di=get_arabic_date(req.date); sid=session_id(req.mosque,req.prayer,req.date)

    def build_rakah(r):
        if not r: return None
        return {"surah":r.surah,"surah_name":r.surah_name,"ayah_start":r.ayah_start,"ayah_end":r.ayah_end,
                "quran_url":r.quran_url or quran_url(r.surah,r.ayah_start,r.ayah_end),
                "confidence":r.confidence or 0.95,"timestamp":r.timestamp or datetime.now().isoformat()}

    session={"id":sid,"mosque":req.mosque,"prayer":req.prayer,"date":req.date,
             "gregorian_date":di["gregorian"],"arabic_date":di["arabic"],"day":di["day"],
             "arabic_day":di["arabic_day"],"hijri":get_hijri_cached(),
             "imam":req.imam or get_imam(req.mosque,req.prayer),
             "rakah_1":build_rakah(req.rakah_1),"rakah_2":build_rakah(req.rakah_2),
             "rakah_2b":build_rakah(req.rakah_2b),
             "confirmed":True,"auto_detected":False}

    upsert_session(session)

    def _fmt(r):
        if not r: return "—"
        return f"{r.get('surah_name','?')} {r.get('ayah_start','?')}–{r.get('ayah_end','?')}"
    print(f" 🌱 SEEDED {req.mosque} {req.prayer} {req.date} → "
          f"R1: {_fmt(session['rakah_1'])}  |  R2: {_fmt(session['rakah_2'])}"
          + (f"  |  R2b: {_fmt(session['rakah_2b'])}" if session.get('rakah_2b') else "")
          + f"  ·  imam: {session.get('imam') or '—'}")

    PRAYER_ORDER={"Fajr":0,"Jumuah":1,"Dhuhr":1,"Asr":2,"Maghrib":3,"Isha":4}
    cur=latest.get(req.mosque,{})
    cur_date=(cur.get("rakah_1") or {}).get("timestamp","")[:10]
    if (req.date>cur_date or
        (req.date==cur_date and PRAYER_ORDER.get(req.prayer,0)>=PRAYER_ORDER.get(cur.get("prayer",""),0))):
        imam=session.get("imam")
        latest[req.mosque]={"prayer":req.prayer,"imam":imam,
                             "rakah_1":{**session["rakah_1"],"imam":imam} if session["rakah_1"] else None,
                             "rakah_2":{**session["rakah_2"],"imam":imam} if session["rakah_2"] else None,
                             "timestamp":(session.get("rakah_1") or {}).get("timestamp"),"seeded":True}

    return {"status":"seeded","id":sid,"total":len(sessions)}

@app.post("/confirm/{sid}")
def confirm_session(sid: str):
    with _sessions_lock:
        idx=next((i for i,s in enumerate(sessions) if s.get("id")==sid),None)
        if idx is None: return {"status":"not_found"}
        sessions[idx]={**sessions[idx],"confirmed":True}
        save_sessions()
        s=sessions[idx]
    print(f" ✅ Confirmed {sid}")
    return {"status":"confirmed","session_id":sid}

@app.get("/pending")
def get_pending():
    all_s=get_all_sessions()
    pending=[{"id":s.get("id"),"mosque":s.get("mosque"),"prayer":s.get("prayer"),
              "date":s.get("date"),"imam":s.get("imam"),"r1":s.get("rakah_1"),"r2":s.get("rakah_2")}
             for s in all_s if s.get("auto_detected") and not s.get("confirmed")]
    return {"pending":pending,"count":len(pending)}

@app.get("/now")
def now(): return latest

@app.get("/makkah")
def makkah_now(): return latest["makkah"]

@app.get("/madinah")
def madinah_now(): return latest["madinah"]

@app.get("/times")
def times():
    mt=get_prayer_times("Makkah"); md=get_prayer_times("Madinah")
    return {"makkah":mt,"madinah":md,"next_prayer":get_next_prayer(mt),"hijri":get_hijri_cached()}

@app.get("/history")
def get_history():
    return sorted(get_all_sessions(),key=lambda x:(x.get("date",""),x.get("mosque",""),x.get("prayer","")),reverse=True)

@app.get("/history/dates")
def get_history_dates():
    dates={}; PRAYER_ORDER={"Fajr":0,"Jumuah":1,"Dhuhr":1,"Asr":2,"Maghrib":3,"Isha":4}
    for s in get_all_sessions():
        d=s["date"]
        if d not in dates:
            dates[d]={"date":d,"gregorian":s.get("gregorian_date",d),"arabic":s.get("arabic_date",""),
                      "day":s.get("day",""),"arabic_day":s.get("arabic_day",""),"hijri":s.get("hijri"),"entries":[]}
        imam=s.get("imam")
        for rn,rk in [(1,"rakah_1"),(2,"rakah_2")]:
            r=s.get(rk)
            if not r: continue
            dates[d]["entries"].append({"mosque":s["mosque"],"prayer":s["prayer"],"rakah":rn,
                "surah":r.get("surah"),"surah_name":r.get("surah_name"),"ayah_start":r.get("ayah_start"),
                "ayah_end":r.get("ayah_end"),"confidence":r.get("confidence",0.95),"imam":imam,
                "quran_url":r.get("quran_url"),"timestamp":r.get("timestamp",d+"T00:00:00"),
                "date":d,"gregorian_date":s.get("gregorian_date",d),"arabic_date":s.get("arabic_date",""),
                "day":s.get("day",""),"arabic_day":s.get("arabic_day",""),"hijri":s.get("hijri")})
        dates[d]["entries"].sort(key=lambda e:(PRAYER_ORDER.get(e["prayer"],9),e["rakah"]))
    return sorted(dates.values(),key=lambda x:x["date"],reverse=True)

@app.get("/imam")
def imam_schedule():
    return {"makkah":{p:IMAM_SCHEDULE.get(("makkah",p)) for p in ["Fajr","Maghrib","Isha","Jumuah"]},
            "madinah":{p:IMAM_SCHEDULE.get(("madinah",p)) for p in ["Fajr","Maghrib","Isha","Jumuah"]}}

@app.get("/servertime")
def servertime():
    return {"utc":datetime.utcnow().strftime("%H:%M"),"saudi":get_saudi_now().strftime("%H:%M"),
            "date":get_arabic_date(),"hijri":get_hijri_cached()}

@app.post("/notify/{sid}")
def notify_session(sid: str):
    all_s = get_all_sessions()
    s = next((x for x in all_s if x.get("id") == sid), None)
    if not s: return {"status": "not_found", "id": sid}
    push_all(s)
    return {"status": "sent", "id": sid}

@app.get("/jumuah")
def get_jumuah():
    return {
        "week":    JUMUAH_CONFIG["week"],
        "hijri":   JUMUAH_CONFIG["hijri"],
        "makkah":  {
            "imam":  JUMUAH_CONFIG["makkah"]["imam"],
            "photo": f"https://huggingface.co/datasets/Ittiba/imams/resolve/main/{JUMUAH_CONFIG['makkah']['imam'].replace(' ','_')}.png?nocache={int(time.time())}",
            "links": JUMUAH_CONFIG["makkah"]["links"],
        },
        "madinah": {
            "imam":  JUMUAH_CONFIG["madinah"]["imam"],
            "photo": f"https://huggingface.co/datasets/Ittiba/imams/resolve/main/{JUMUAH_CONFIG['madinah']['imam'].replace(' ','_')}.png?nocache={int(time.time())}",
            "links": JUMUAH_CONFIG["madinah"]["links"],
        },
        "fallback": JUMUAH_CONFIG["fallback"],
    }

class JumuahLinks(BaseModel):
    arabic: Optional[str] = ""; english: Optional[str] = ""; urdu: Optional[str] = ""
    french: Optional[str] = ""; indonesian: Optional[str] = ""; turkish: Optional[str] = ""
    malay: Optional[str] = ""; russian: Optional[str] = ""; chinese: Optional[str] = ""
    hausa: Optional[str] = ""; farsi: Optional[str] = ""; hindi: Optional[str] = ""; bengali: Optional[str] = ""

class JumuahMosque(BaseModel):
    imam: str; links: JumuahLinks

class JumuahUpdate(BaseModel):
    week: str; hijri: str; makkah: JumuahMosque; madinah: JumuahMosque

class ImamScheduleUpdate(BaseModel):
    makkah: dict; madinah: dict

@app.post("/admin/jumuah")
def update_jumuah(req: JumuahUpdate):
    global JUMUAH_CONFIG
    JUMUAH_CONFIG["week"]=req.week; JUMUAH_CONFIG["hijri"]=req.hijri
    JUMUAH_CONFIG["makkah"]["imam"]=req.makkah.imam; JUMUAH_CONFIG["makkah"]["links"]=req.makkah.links.dict()
    JUMUAH_CONFIG["madinah"]["imam"]=req.madinah.imam; JUMUAH_CONFIG["madinah"]["links"]=req.madinah.links.dict()
    print(f"✓ Jumuah updated: {req.week} — Makkah: {req.makkah.imam} · Madinah: {req.madinah.imam}")
    save_jumuah_config()
    return {"status":"ok","week":req.week}

@app.post("/admin/imam_schedule")
def update_imam_schedule(req: ImamScheduleUpdate):
    global IMAM_SCHEDULE
    prayers = ["Fajr","Maghrib","Isha","Jumuah"]
    for mosque in ["makkah","madinah"]:
        src = req.makkah if mosque=="makkah" else req.madinah
        for prayer in prayers:
            name = src.get(prayer,"")
            if name: IMAM_SCHEDULE[(mosque,prayer)] = name
    print(f"✓ Imam schedule updated")
    save_imam_schedule()
    return {"status":"ok","schedule":{
        "makkah":{p:IMAM_SCHEDULE.get(("makkah",p)) for p in prayers},
        "madinah":{p:IMAM_SCHEDULE.get(("madinah",p)) for p in prayers},
    }}

@app.post("/admin/upload_photo")
async def upload_photo(request: Request):
    from huggingface_hub import HfApi
    from io import BytesIO
    import base64 as b64lib
    body = await request.json()
    filename = body.get("filename","").strip(); content = body.get("content","")
    if not filename or not content: return {"status":"error","message":"filename and content required"}
    if not filename.endswith(".png") or "/" in filename or ".." in filename:
        return {"status":"error","message":"invalid filename"}
    try:
        img_bytes = b64lib.b64decode(content)
        api = HfApi(token=hf_token)
        api.upload_file(path_or_fileobj=BytesIO(img_bytes),path_in_repo=filename,
            repo_id="Ittiba/imams",repo_type="dataset",commit_message=f"Upload imam photo: {filename}")
        print(f"✓ Uploaded imam photo: {filename} ({len(img_bytes)} bytes)")
        return {"status":"ok","filename":filename,"size":len(img_bytes)}
    except Exception as e:
        print(f"✗ Photo upload error: {e}"); return {"status":"error","message":str(e)}

@app.get("/trigger_prayer/{mosque}/{prayer}")
def trigger(mosque:str, prayer:str):
    threading.Thread(target=smart_listen,args=(mosque,prayer)).start()
    return {"status":"started","mosque":mosque,"prayer":prayer}

@app.get("/test/{mosque}/{prayer}")
def test_silent(mosque: str, prayer: str):
    """
    Silent test — runs full detection pipeline (record → parse → vote) but
    does NOT save, does NOT push any notifications. Watch Space logs for results.
    """
    def run():
        print(f"\n🧪 SILENT TEST: {mosque.upper()} {prayer}")
        try:
            classified = record_prayer(mosque, prayer)
            r1, r2 = parse_rakahs(classified, prayer)

            print(f"\n🧪 === R1 ({len(r1)} chunks) ===")
            r1_result = vote_surah(r1, prayer_name=prayer)
            if r1_result:
                print(f"🧪 R1: {r1_result['surah_name']} {r1_result['ayah']}–{r1_result['ayah_end']} (conf={r1_result['confidence']})")
            else:
                print("🧪 R1: not detected")

            print(f"\n🧪 === R2 ({len(r2)} chunks) ===")
            r1_surah = r1_result["surah"] if r1_result else None
            r2_result = vote_surah(r2, continuity_id=r1_surah, prayer_name=prayer,
                                   r1_ayah_end=r1_result["ayah_end"] if r1_result else 0)
            if r2_result:
                print(f"🧪 R2: {r2_result['surah_name']} {r2_result['ayah']}–{r2_result['ayah_end']} (conf={r2_result['confidence']})")
            else:
                print("🧪 R2: not detected")

            print("🧪 TEST COMPLETE — nothing saved, no notifications sent")
        except Exception as e:
            import traceback
            print(f"🧪 ERROR: {e}"); traceback.print_exc()

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started", "note": "watch Space logs — nothing will be saved or pushed"}

# ── Quicktest (background + poll — no synchronous timeout) ────────────────────
_quicktest_result = {"status": "idle"}
_quicktest_lock = threading.Lock()

def _run_quicktest(mosque, secs):
    quran = get_quran()
    chunk_secs = 20
    classified = []
    chunk_log = []
    elapsed = 0
    t_start = time.time()
    try:
        while elapsed < secs:
            audio = sample_audio(mosque, secs=chunk_secs)
            if audio is None:
                elapsed += chunk_secs
                continue
            for i in range(0, len(audio), 16000 * chunk_secs):
                chunk = audio[i:i + 16000 * chunk_secs]
                if len(chunk) < 16000 * 3:
                    continue
                text = transcribe(chunk)
                cls = classify(text)
                classified.append((chunk, text, cls))

                # Normalize both sides before matching, same as vote_surah does —
                # without this, quicktest (the debug tool for diagnosing exactly
                # this kind of matching problem) silently scores every match
                # worse than production ever would, making its output misleading
                # for the one thing it exists to show you.
                best = None
                if cls == "quran" and len(text.strip().split()) > 2:
                    ntext_qt = strip_diacritics(text)
                    for surah in quran:
                        if surah["id"] == 1:
                            continue
                        for ayah in surah["verses"]:
                            s = SequenceMatcher(None, ntext_qt, ayah.get("_norm", strip_diacritics(ayah["text"]))).ratio()
                            if best is None or s > best["conf"]:
                                best = {"surah": surah["id"],
                                        "name": surah["transliteration"],
                                        "ayah": ayah["id"],
                                        "conf": round(s, 3)}
                chunk_log.append({
                    "t": elapsed,
                    "cls": cls,
                    "text": text,
                    "best_match": best,
                    "fatihah_sim": round(sim(text, FATIHAH_PHRASES), 3),
                    "salam_sim": round(sim(text, SALAM_PHRASES), 3),
                })
                with _quicktest_lock:
                    _quicktest_result["chunks"] = list(chunk_log)
                    _quicktest_result["n_chunks"] = len(classified)
            del audio
            elapsed += chunk_secs

        r1, r2 = parse_rakahs(classified, "Isha")
        r1_result = vote_surah(r1, prayer_name="Isha")
        r2_result = vote_surah(
            r2,
            continuity_id=(r1_result["surah"] if r1_result else None),
            prayer_name="Isha",
            r1_ayah_end=(r1_result["ayah_end"] if r1_result else 0),
        )

        def fmt(r):
            if not r:
                return None
            return {"surah": r["surah"], "surah_name": r["surah_name"],
                    "ayah_start": r["ayah"], "ayah_end": r["ayah_end"],
                    "confidence": r["confidence"]}

        with _quicktest_lock:
            _quicktest_result.update({
                "status": "done",
                "mosque": mosque,
                "recorded_secs": secs,
                "actual_secs": round(time.time() - t_start),
                "n_chunks": len(classified),
                "cls_counts": {c: sum(1 for _, _, x in classified if x == c)
                               for c in ("fatihah", "quran", "takbeer", "salam", "empty")},
                "r1": fmt(r1_result),
                "r2": fmt(r2_result),
                "chunks": chunk_log,
            })
        print("🧪 QUICKTEST COMPLETE — nothing saved, no notifications sent")
    except Exception as e:
        import traceback
        traceback.print_exc()
        with _quicktest_lock:
            _quicktest_result.update({"status": "error", "error": str(e)})

@app.get("/quicktest/{mosque}")
def quick_test(mosque: str, secs: int = 120):
    if mosque not in ("makkah", "madinah"):
        return {"error": "mosque must be 'makkah' or 'madinah'"}
    secs = max(20, min(secs, 600))
    with _quicktest_lock:
        if _quicktest_result.get("status") == "running":
            return {"status": "already_running", "note": "poll /quicktest_result"}
        _quicktest_result.clear()
        _quicktest_result.update({"status": "running", "mosque": mosque,
                                  "recorded_secs": secs, "n_chunks": 0, "chunks": []})
    threading.Thread(target=_run_quicktest, args=(mosque, secs), daemon=True).start()
    return {"status": "started", "mosque": mosque, "secs": secs,
            "note": f"recording ~{secs}s — poll /quicktest_result every few seconds"}

@app.get("/quicktest_result")
def quick_test_result():
    with _quicktest_lock:
        return dict(_quicktest_result)

@app.get("/selftest")
def selftest():
    quran = get_quran()

    cases = [
        (16, 90, 93), (31, 13, 17), (7, 178, 180), (18, 107, 110),
        (2, 255, 257), (36, 1, 5), (67, 1, 4), (112, 1, 4),
        (55, 1, 6), (9, 40, 42),
    ]

    def garble(s):
        s = (s.replace("أ","ا").replace("إ","ا").replace("آ","ا")
               .replace("ة","ه").replace("ى","ي").replace("ؤ","و").replace("ئ","ي"))
        ws = s.split()
        ws = [w for i, w in enumerate(ws) if i % 6 != 5]
        return " ".join(ws)

    results = []; correct = 0
    for sid, a1, a2 in cases:
        sd = next((x for x in quran if x["id"] == sid), None)
        if not sd:
            continue
        chunks = [(None, garble(v["text"])) for v in sd["verses"] if a1 <= v["id"] <= a2]
        if not chunks:
            continue
        r = vote_surah(chunks, prayer_name="Isha")
        got = r["surah"] if r else None
        ok = (got == sid)
        correct += ok
        results.append({
            "expected_surah": sid, "expected_name": sd["transliteration"],
            "expected_ayahs": f"{a1}-{a2}",
            "got_surah": got,
            "got_name": r["surah_name"] if r else None,
            "got_ayahs": f"{r['ayah']}-{r['ayah_end']}" if r else None,
            "confidence": r["confidence"] if r else None,
            "pass": ok,
        })
    return {
        "test": "matcher self-test (normalized matching, garbled known ayahs)",
        "score": f"{correct}/{len(results)}",
        "accuracy": round(correct / max(len(results), 1), 2),
        "cases": results,
    }

if __name__=="__main__":
    import uvicorn
    uvicorn.run("api:app",host="0.0.0.0",port=int(os.environ.get("PORT",7860)))
