import os, uuid, json, shutil, subprocess, re
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

ROOT = Path(__file__).parent
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)
OUT = ROOT / "exports"
OUT.mkdir(exist_ok=True)
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
app = FastAPI(title="KAI VIRAL AI v2")
app.mount("/exports", StaticFiles(directory=str(OUT)), name="exports")
model = None

def get_model():
    global model
    if model is None:
        # CPU mode is the most portable free setup. First run downloads model weights.
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return model

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2500:])
    return p.stdout

def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]", "_", name or "video.mp4")

def get_duration(path):
    s = run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)])
    return float(s.strip())

def transcribe(path):
    segments, info = get_model().transcribe(str(path), language="en", vad_filter=True, word_timestamps=True)
    result=[]
    for seg in segments:
        result.append({"start":float(seg.start),"end":float(seg.end),"text":seg.text.strip()})
    return result

def scene_scores(path, duration, transcript):
    # Scene boundaries from visual changes + speech density. This is a heuristic, not a guarantee of virality.
    video = open_video(str(path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=24.0, min_scene_len=15))
    sm.detect_scenes(video)
    scenes = sm.get_scene_list()
    ranges=[]
    if scenes:
        for a,b in scenes:
            ranges.append((a.get_seconds(), b.get_seconds()))
    else:
        step=20
        ranges=[(x,min(x+step,duration)) for x in np.arange(0,duration,step)]
    scored=[]
    for start,end in ranges:
        if end-start < 8: continue
        words = sum(len(x["text"].split()) for x in transcript if x["start"] < end and x["end"] > start)
        text = " ".join(x["text"] for x in transcript if x["start"] < end and x["end"] > start).lower()
        excitement = sum(text.count(w) for w in ["wow","no way","insane","crazy","oh my","wait","what","amazing","bro"])
        # Slight preference for 20-45s windows, with speech and expressive phrases.
        score = min(words/30, 2.0) + excitement*0.35 + (0.5 if 18 <= end-start <= 50 else 0)
        scored.append({"start":start,"end":end,"score":score})
    # Merge/expand candidates to useful Shorts lengths.
    candidates=[]
    for item in sorted(scored,key=lambda x:x["score"],reverse=True):
        s=max(0,item["start"]-3)
        e=min(duration,max(item["end"]+4,s+35))
        if e-s < 15: e=min(duration,s+25)
        if all(min(e,c["end"])-max(s,c["start"]) < 12 for c in candidates):
            candidates.append({"start":s,"end":e,"score":item["score"]})
        if len(candidates)>=12: break
    return sorted(candidates,key=lambda x:x["score"],reverse=True)

def make_hook(transcript, start, end):
    text=" ".join(x["text"] for x in transcript if x["start"] < end and x["end"] > start)
    text=text.strip()
    if text:
        phrase=" ".join(text.split()[:9]).upper()
        return "WAIT FOR IT: " + phrase
    return "YOU WON'T BELIEVE WHAT HAPPENS NEXT!"

def make_ending(transcript, start, end):
    return "WOULD YOU TRY THIS? COMMENT BELOW!"

def srt_time(sec):
    ms=int(sec*1000); h=ms//3600000; ms%=3600000; m=ms//60000; ms%=60000; s=ms//1000; ms%=1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_srt(transcript, start, end, dest):
    lines=[]; idx=1
    for seg in transcript:
        a=max(seg["start"],start); b=min(seg["end"],end)
        if b<=a: continue
        txt=seg["text"].replace("-->","")
        lines += [str(idx),f"{srt_time(a-start)} --> {srt_time(b-start)}",txt,""]
        idx+=1
    Path(dest).write_text("\n".join(lines),encoding="utf-8")

def crop_filter(path):
    # Face-aware crop: detect face positions in sampled frames; choose a stable horizontal center.
    cap=cv2.VideoCapture(str(path))
    cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    centers=[]; n=0
    while n<90:
        ok,frame=cap.read()
        if not ok: break
        if n%12==0:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=cascade.detectMultiScale(gray,1.1,4,minSize=(40,40))
            if len(faces):
                x,y,w,h=max(faces,key=lambda f:f[2]*f[3])
                centers.append((x+w/2)/frame.shape[1])
        n+=1
    cap.release()
    if centers: return float(np.median(centers))
    return 0.5

def render_clip(source, start, end, srt, output, center):
    # Scale to fill 1080x1920; crop around detected face center; burn English captions.
    dur=end-start
    escaped=str(srt).replace("\\","/").replace(":","\\:").replace("'","\\'")
    vf=(f"crop='min(ih*9/16,iw)':'min(iw*16/9,ih)':"
        f"'max(0,min(iw-ow,{center}*iw-ow/2))':'(ih-oh)/2',"
        "scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,")
    # Caption style is readable and high-contrast.
    vf += f"subtitles='{escaped}':force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=1,Alignment=2,MarginV=150'"
    run(["ffmpeg","-y","-ss",str(start),"-i",str(source),"-t",str(dur),"-vf",vf,
         "-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k",
         "-movflags","+faststart",str(output)])

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT/"index.html").read_text(encoding="utf-8")

@app.post("/inspect-url")
def inspect_url(url: str = Form(...)):
    # Metadata inspection only. This intentionally does not download YouTube media.
    if not re.match(r"^https?://(www\.)?(youtube\.com|youtu\.be)/",url):
        raise HTTPException(400,"Please enter a valid YouTube URL.")
    try:
        r=requests.get("https://www.youtube.com/oembed",params={"url":url,"format":"json"},timeout=15)
        r.raise_for_status()
        d=r.json()
        return {"title":d.get("title"),"author":d.get("author_name"),"thumbnail":d.get("thumbnail_url"),
                "message":"URL recognized. For editing, upload a video file you own or are authorized to use."}
    except Exception:
        return {"message":"URL accepted, but metadata could not be read. Upload an authorized source video to continue."}

@app.post("/generate")
async def generate(file: UploadFile = File(...), shorts_count: int = Form(3)):
    ext=Path(file.filename or "").suffix.lower()
    if ext not in [".mp4",".mov",".mkv",".webm",".m4v"]:
        raise HTTPException(400,"Upload MP4, MOV, MKV, WEBM or M4V.")
    if shorts_count<1 or shorts_count>10: raise HTTPException(400,"Choose 1–10 Shorts.")
    job=uuid.uuid4().hex[:10]
    jobdir=WORK/job; jobdir.mkdir()
    source=jobdir/safe_name(file.filename)
    with source.open("wb") as f: shutil.copyfileobj(file.file,f)
    try:
        duration=get_duration(source)
        if duration>4*3600: raise HTTPException(400,"Maximum input length is 4 hours.")
        transcript=transcribe(source)
        candidates=scene_scores(source,duration,transcript)[:shorts_count]
        if not candidates: raise HTTPException(400,"Could not find suitable scenes in this video.")
        center=crop_filter(source)
        results=[]
        for i,c in enumerate(candidates,1):
            srt=jobdir/f"short_{i}.srt"
            mp4=OUT/f"{job}_{i}.mp4"
            write_srt(transcript,c["start"],c["end"],srt)
            render_clip(source,c["start"],c["end"],srt,mp4,center)
            results.append({"file":f"/exports/{mp4.name}","start":round(c["start"],1),"end":round(c["end"],1),
                            "hook":make_hook(transcript,c["start"],c["end"]),
                            "ending":make_ending(transcript,c["start"],c["end"])})
        return {"job":job,"caption_language":"English","shorts":results,
                "note":"Moment ranking is heuristic. Review clips, captions and rights before publishing."}
    except HTTPException: raise
    except Exception as e: raise HTTPException(500,str(e))
