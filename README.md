# YOUTUBE-AUTOMATION
AI youtube automation generation
#!/usr/bin/env python3
"""
render_video.py
================
Runs inside a GitHub Action. Reads a job payload (script + scene prompts +
metadata) that n8n sent via `repository_dispatch`, then:

  1. Generates one illustration per scene with Pollinations.ai (free, no key)
  2. Generates narration audio per scene with the free StreamElements TTS
     endpoint (no key, no signup)
  3. Assembles a Ken-Burns style video with ffmpeg (tested pipeline)
  4. Burns in captions
  5. Builds a thumbnail
  6. Uploads the result to YouTube via the YouTube Data API v3

Every external call is wrapped in retries because the free endpoints this
relies on (Pollinations / StreamElements) are unauthenticated community
services with no uptime SLA — expect occasional retries/failures in
production and check the Action logs.
"""

import os
import json
import time
import textwrap
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

FPS = 25
RES = "1920x1080"
THUMB_RES = (1280, 720)
WORKDIR = Path("render_work")
FONT_BOLD = os.environ.get("FONT_BOLD", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
STREAMELEMENTS_VOICE = os.environ.get("STREAMELEMENTS_VOICE", "Brian")
MAX_TTS_CHARS = 480  # StreamElements free endpoint truncates/fails on long strings


# --------------------------------------------------------------------------- #
# Input payload
# --------------------------------------------------------------------------- #
def load_payload() -> dict:
    """Reads the repository_dispatch client_payload off the GitHub event."""
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and Path(event_path).exists():
        event = json.loads(Path(event_path).read_text())
        payload = event.get("client_payload")
        if payload:
            return payload
    # Local/manual testing fallback
    local = Path("payload.json")
    if local.exists():
        return json.loads(local.read_text())
    raise SystemExit("No payload found (expected client_payload or payload.json)")


# --------------------------------------------------------------------------- #
# Free asset fetchers
# --------------------------------------------------------------------------- #
def fetch_image(prompt: str, out_path: Path, retries: int = 3):
    style = ", cinematic historical illustration, dramatic lighting, painterly, no text, no watermark"
    encoded = urllib.parse.quote(prompt + style)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&nologo=true"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return
        except Exception as e:
            print(f"  [image] attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    raise RuntimeError(f"Could not fetch image for prompt: {prompt[:60]}")


def _streamelements_chunk(text: str, out_path: Path):
    url = "https://api.streamelements.com/kappa/v2/speech"
    params = {"voice": STREAMELEMENTS_VOICE, "text": text}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    if len(r.content) < 500:
        raise RuntimeError("TTS response too small, likely an error payload")
    out_path.write_bytes(r.content)


def fetch_tts(text: str, out_path: Path, retries: int = 3):
    """Free TTS via StreamElements. Chunks long narration and concatenates."""
    chunks = textwrap.wrap(text, MAX_TTS_CHARS, break_long_words=False) or [text]
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        cpath = out_path.with_suffix(f".part{i}.mp3")
        for attempt in range(retries):
            try:
                _streamelements_chunk(chunk, cpath)
                chunk_paths.append(cpath)
                break
            except Exception as e:
                print(f"  [tts] chunk {i} attempt {attempt + 1} failed: {e}")
                time.sleep(3)
        else:
            raise RuntimeError(f"Could not synthesize TTS chunk: {chunk[:60]}")

    if len(chunk_paths) == 1:
        chunk_paths[0].rename(out_path)
        return

    list_file = out_path.with_suffix(".concat.txt")
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in chunk_paths))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(out_path)],
        check=True, capture_output=True,
    )


# --------------------------------------------------------------------------- #
# Video assembly (ffmpeg pipeline — validated against synthetic assets)
# --------------------------------------------------------------------------- #
def get_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def build_scene_clip(image_path: Path, audio_path: Path, out_path: Path) -> float:
    duration = get_duration(audio_path)
    frames = max(1, round(duration * FPS))
    vf = (
        f"scale=1920:1080:force_original_aspect_ratio=increase,"
        f"crop=1920:1080,"
        f"zoompan=z='min(zoom+0.0012,1.15)':d={frames}:s={RES}:fps={FPS},"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", str(image_path), "-i", str(audio_path),
        "-filter_complex", f"[0:v]{vf}[v]",
        "-map", "[v]", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac",
        "-t", str(duration), "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return duration


def concat_scenes(scene_paths, out_path: Path):
    list_file = WORKDIR / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{p.resolve()}'" for p in scene_paths))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", str(out_path)],
        check=True, capture_output=True,
    )


def make_srt(scenes_with_durations, out_path: Path):
    def fmt(t):
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02}:{int(m):02}:{s:06.3f}".replace(".", ",")

    t, lines = 0.0, []
    for i, (text, dur) in enumerate(scenes_with_durations, start=1):
        lines += [str(i), f"{fmt(t)} --> {fmt(t + dur)}", text, ""]
        t += dur
    out_path.write_text("\n".join(lines))


def burn_captions(video_path: Path, srt_path: Path, out_path: Path):
    style = ("FontName=DejaVu Sans,FontSize=22,PrimaryColour=&H00FFFFFF,"
             "OutlineColour=&H00000000,BorderStyle=3,Outline=2,MarginV=60")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path),
         "-vf", f"subtitles={srt_path}:force_style='{style}'",
         "-c:a", "copy", str(out_path)],
        check=True, capture_output=True,
    )


# --------------------------------------------------------------------------- #
# Thumbnail
# --------------------------------------------------------------------------- #
def make_thumbnail(src: Path, out_path: Path, title: str):
    img = Image.open(src).convert("RGB").resize(THUMB_RES)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, 460, THUMB_RES[0], THUMB_RES[1]], fill=(0, 0, 0, 160))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    font_size = 64
    font = ImageFont.truetype(FONT_BOLD, font_size)
    words, lines, line = title.split(), [], ""
    for w in words:
        test = (line + " " + w).strip()
        if draw.textlength(test, font=font) > THUMB_RES[0] - 100:
            lines.append(line)
            line = w
        else:
            line = test
    lines.append(line)

    y = 480
    for ln in lines[:3]:
        draw.text((50, y), ln, font=font, fill="white", stroke_width=3, stroke_fill="black")
        y += font_size + 10
    img.save(out_path, quality=92)


# --------------------------------------------------------------------------- #
# YouTube upload
# --------------------------------------------------------------------------- #
def upload_to_youtube(video_path: Path, thumb_path: Path, title: str, description: str,
                       tags: list, privacy_status: str = "private") -> str:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    creds = Credentials(
        None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"],
        client_id=os.environ["YT_CLIENT_ID"],
        client_secret=os.environ["YT_CLIENT_SECRET"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    youtube = build("youtube", "v3", credentials=creds)

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:30],
            "categoryId": "27",  # Education; switch to "22" People&Blogs etc. if preferred
        },
        "status": {"privacyStatus": privacy_status, "selfDeclaredMadeForKids": False},
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"  upload progress: {int(status.progress() * 100)}%")
    video_id = response["id"]

    youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(thumb_path))).execute()
    print(f"Uploaded: https://youtu.be/{video_id}")
    return video_id


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    WORKDIR.mkdir(exist_ok=True)
    payload = load_payload()

    title = payload["title"]
    description = payload.get("description", "")
    tags = payload.get("tags", [])
    privacy_status = payload.get("privacy_status", os.environ.get("PRIVACY_STATUS", "private"))
    scenes = payload["scenes"]

    clip_paths, durations = [], []
    first_image = None

    for i, scene in enumerate(scenes):
        print(f"Scene {i + 1}/{len(scenes)}")
        img_path = WORKDIR / f"scene_{i}.jpg"
        aud_path = WORKDIR / f"scene_{i}.mp3"
        clip_path = WORKDIR / f"clip_{i}.mp4"

        fetch_image(scene["image_prompt"], img_path)
        fetch_tts(scene["narration"], aud_path)
        dur = build_scene_clip(img_path, aud_path, clip_path)

        clip_paths.append(clip_path)
        durations.append((scene["narration"], dur))
        if first_image is None:
            first_image = img_path

    joined = WORKDIR / "joined.mp4"
    concat_scenes(clip_paths, joined)

    srt_path = WORKDIR / "captions.srt"
    make_srt(durations, srt_path)

    final_video = WORKDIR / "final.mp4"
    burn_captions(joined, srt_path, final_video)

    thumb_path = WORKDIR / "thumbnail.jpg"
    make_thumbnail(first_image, thumb_path, title)

    video_id = upload_to_youtube(final_video, thumb_path, title, description, tags, privacy_status)
    print(json.dumps({"video_id": video_id, "url": f"https://youtu.be/{video_id}"}))


if __name__ == "__main__":
    main()
