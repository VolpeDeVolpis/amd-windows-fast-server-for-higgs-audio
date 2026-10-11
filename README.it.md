# Server veloce per Higgs Audio v3 su schede AMD (ROCm su Windows), senza ComfyUI

> **Credito.** Questo progetto usa **Higgs Audio v3** di **[Boson AI](https://www.boson.ai/higgs-audio)**. È uno strumento indipendente e non ufficiale, *Derived from Higgs TTS 3, licensed from Boson AI USA, Inc.* **Non contiene i pesi del modello.** Vedi [Licenza](#licenza-da-leggere-per-prima) e [NOTICE](NOTICE).
>
> English: [README.md](README.md)

Un piccolo server locale (`POST /tts` -> WAV) che genera voce con Higgs Audio v3 su una **scheda AMD Radeon con ROCm su Windows**, molto più veloce dello stesso modello dentro ComfyUI. Tutto gira sulla scheda video, niente sul processore.

## Risultati misurati

Stesso testo (74 s di voce), stessa voce clonata, stessa scheda (AMD Radeon RX 7900 XTX, 24 GB), Windows, 11 ott 2026:

| Sistema | Tempo | Velocità |
|---|---|---|
| ComfyUI (nodo TTS-Audio-Suite) | 446 s | 1x |
| Questo server, motore torch, paragrafi interi (scheda pulita) | 129 s | 3,5x |
| **Questo server, motore llama.cpp (predefinito)** | **17 s** (accensione ~32 s, poi si spegne da solo) | **~26x** |

Altre misure (stessa macchina):
- Costo per passo: 13 ms con 1 testo, 18 ms con 8, 28 ms con 12 insieme (motore torch: 200-330 ms).
- Il "cervello" (parte Qwen3) convertito in GGUF coincide col modello originale al 99,999%.
- Conversione in audio con MIOpen spenta: 0,2 s invece di 9,6 s per 30 s di voce, risultato identico (differenza massima 0,00001).
- Picco di memoria video 14,7 GB (motore llama.cpp), memoria condivisa +0,2 GB (niente traboccamento nella RAM).
- Con il motore torch il tempo varia 129-203 s secondo il carico del processore (usa un nucleo intero).

Sono numeri di una sola macchina e di un solo testo: i tuoi saranno diversi.

## Perché ComfyUI era lento (misurato)
1. **Mille piccoli comandi a ogni passo.** Ogni passo manda ~1000 comandi alla scheda e su Windows ognuno ha un costo fisso: per questo la scheda sembrava ferma (0-2%). Rimedio: generare **più paragrafi insieme nello stesso passo**. Con 4 testi il costo del passo cresce solo del 15%; con 16 la resa è 7 volte maggiore.
2. **Conversione finale in audio.** La libreria MIOpen di AMD, su Windows, sceglie metodi di ripiego lentissimi. Spegnendola (`torch.backends.cudnn.enabled = False`) l'audio è lo stesso e molto più veloce. *Il trucco si può applicare anche in ComfyUI.*
3. La conversione rallenta molto sui pezzi lunghi: va fatta un pezzo alla volta.
4. Spostare il "cervello" Qwen3 in llama.cpp (un comando grosso per passo invece di ~1000) dà il resto del guadagno.

## Come funziona
- Riceve il testo, tiene interi i paragrafi (taglia a fine frase solo oltre 500 caratteri), genera fino a 12 pezzi insieme, li unisce con 0,25 s di pausa e ripulisce le giunture (code di rumore prima delle pause, "colpi" ai bordi).
- Il transformer gira in llama.cpp; scelta dei suoni e conversione audio restano in torch sulla scheda.
- `src/higgs_client.py` può accendere il server quando serve: controlla prima la memoria video libera (se non riesce a misurarla non parte); il server si spegne da solo dopo 2 minuti senza lavoro.
- API: `POST /tts` `{"text","voice","seed"}` -> WAV; `GET /voices`; `GET /health`. Ascolta solo su `127.0.0.1:8090`.

## Requisiti
- Windows 10/11, scheda AMD Radeon supportata da ROCm per Windows, **>= 18 GB di memoria video liberi** consigliati (provato su 24 GB; picco ~15 GB). Altre schede non provate.
- Python 3.12 con **PyTorch ROCm per Windows** (provato: torch 2.12.0+rocm7.15.0a20260709, torchaudio 2.11.0, transformers 5.13.0) più `numpy`, `soundfile`, `safetensors`, `tokenizers`, `gguf`.
- I file del modello Higgs Audio v3 di Boson (da scaricare, vedi sotto).
- Una copia di `native.py` dal nodo ComfyUI **TTS-Audio-Suite** (MIT), cartella `engines/higgs_audio_v3/` (lo importiamo, non lo distribuiamo).
- Solo per il motore veloce: la **release ufficiale di llama.cpp HIP (ROCm) per Windows** al commit `e85e15cf6` (il ponte ctypes vale solo per quella versione) e **LM Studio** installato col suo runtime ROCm (fornisce `libhipblas.dll`). Senza, usa `--engine torch` (3,5x invece di ~26x).

## Installazione passo per passo
1. Installa PyTorch ROCm per Windows in un ambiente Python 3.12, poi `pip install numpy soundfile safetensors tokenizers gguf`.
2. **Scarica il modello tu stesso** dalla pagina ufficiale: <https://huggingface.co/bosonai/higgs-audio-v3-tts-4b> (accetta lì la licenza di Boson, per esempio `huggingface-cli download bosonai/higgs-audio-v3-tts-4b --local-dir <cartella>`). **Non** ricaricarlo da nessuna parte. Pagina del prodotto: <https://www.boson.ai/higgs-audio>.
3. Procurati il nodo ComfyUI TTS-Audio-Suite (clone o download): servono `engines/higgs_audio_v3/native.py` e i file del pacchetto accanto.
4. Metti le TUE voci di riferimento in `voices/` (`NOME.wav` + eventuale `NOME.txt` con la trascrizione). Solo voci di cui hai diritti e consenso.
5. Imposta le variabili d'ambiente (PowerShell o impostazioni di sistema):
   ```powershell
   $env:HIGGS_PYTHON      = "C:\percorso\python.exe"
   $env:HIGGS_MODEL_DIR   = "C:\percorso\higgs-audio-v3-tts-4b"
   $env:HIGGS_NATIVE_PATH = "C:\percorso\TTS-Audio-Suite\engines\higgs_audio_v3\native.py"
   $env:HIGGS_VOICES_DIR  = "C:\percorso\voci"            # facoltativo, predefinito .\voices
   $env:LLAMA_CPP_DIR     = "C:\percorso\llama.cpp-hip"   # solo motore veloce
   ```
6. Solo motore veloce: `powershell scripts\make_bridge_dll.ps1` (copia una DLL dal TUO LM Studio), poi converti il modello in locale:
   `python src\llama_engine\convert_backbone.py <HIGGS_MODEL_DIR> src\llama_engine\models\higgs_backbone_bf16.gguf`
   (8 GB; resta sul tuo PC, è un derivato dei pesi di Boson).
7. Controlla che la memoria video sia libera: `powershell scripts\check_vram.ps1`.
8. Avvia: `scripts\start_server.bat` (opzioni: `--engine torch` per la modalità più lenta senza llama.cpp, `--port`, `--batch`, `--idle-shutdown 0` per non spegnersi mai).
9. Prova: `python tools\send_test.py testo.txt LaTuaVoce uscita.wav`.

## Script
`scripts/start_server.bat` avvio; `scripts/check_vram.ps1` memoria video dedicata/condivisa; `scripts/run_with_turn.ps1` esegue un lavoro pesante con file di turno, controllo memoria libera e sorveglianza della memoria condivisa; `scripts/make_bridge_dll.ps1`; `tools/verifica_giunture.py` misura i difetti alle giunture; `tools/bench_*.py`, `profile_step.py`, `snr_codec.py` misure; `tests/` prove solo CPU (`python -m pytest tests`).

## Limiti noti
- Provato su una sola macchina (RX 7900 XTX, Windows). Altre schede e driver ignoti.
- Il ponte con llama.cpp dipende da un commit preciso e da una DLL presa da LM Studio; aggiornando llama.cpp vanno aggiornate le strutture (il server si ferma con un messaggio chiaro).
- Il server usa un nucleo intero del processore; con il processore saturo rallenta.
- I paragrafi oltre 500 caratteri sono tagliati a fine frase (oltre, Higgs si ferma da solo); il taglio sulle virgole creava cambi di tono udibili ed è stato abbandonato.
- I grafi HIP erano più lenti su Windows+AMD e hanno bloccato un processo; codec in fp16/bf16 rompe o altera il suono; SGLang/vLLM funzionano solo su Linux con NVIDIA.
- Alcuni commenti e messaggi nel codice sono ancora in italiano.

## Licenza (da leggere per prima)
- **Higgs Audio v3 non è open source.** Ha la *Boson Higgs TTS 3 Research and Non-Commercial License*: gratuito per ricerca e uso personale/non commerciale; **gratuito per i creatori digitali che pubblicano, anche con guadagno, contenuti fatti con il modello, citando Boson** (sezione II-A); tutto il resto (offrirlo come servizio/API, integrarlo in un prodotto per terzi, ridistribuirlo o distribuire derivati, rivenderlo) richiede una licenza commerciale di Boson (<https://boson.ai>). Vietati anche: clonare o imitare voci senza consenso, inganni, usare gli output per addestrare altri modelli generativi. Il testo completo è sulla pagina del modello.
- **Citazione obbligatoria per audio/video che pubblichi**: *"This audio was created with Boson AI's Higgs Audio — https://www.boson.ai/higgs-audio"*, nell'audio o ben visibile nella descrizione.
- Il vecchio repository GitHub `higgs-audio` di Boson è Apache-2.0 e qui non è usato; il suo README dice che v3 è un rilascio autonomo. I pesi v3 hanno la licenza qui sopra.
- Licenza del codice di questo repository: vedi `LICENSE` (MIT, © 2026 Volpe). Crediti di terzi: [THIRD_PARTY.md](THIRD_PARTY.md). La licenza del codice **non** cambia quella del modello.
- Nessun legame né approvazione da parte di Boson AI, AMD, LM Studio o ComfyUI.
