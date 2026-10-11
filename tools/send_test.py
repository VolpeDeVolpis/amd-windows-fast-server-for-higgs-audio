# Dipendenze: server.py acceso. Manda un testo al server e misura i secondi. Uso: prova_server.py file.txt voce uscita.wav
import json, sys, time, urllib.request
text = open(sys.argv[1], encoding="utf-8-sig").read().strip()
body = json.dumps({"text": text, "voice": sys.argv[2], "seed": 42}).encode("utf-8")
start = time.perf_counter()
request = urllib.request.Request("http://127.0.0.1:8090/tts", data=body, headers={"Content-Type": "application/json"})
audio = urllib.request.urlopen(request, timeout=3600).read()
open(sys.argv[3], "wb").write(audio)
seconds_audio = (len(audio) - 44) / 2 / 24000
print(f"wall_s={time.perf_counter() - start:.1f} audio_s={seconds_audio:.1f}")
