# Vibz MusicGen API

## Overview
A FastAPI-based music generation API using Facebook's MusicGen model. Generates 20-45 second instrumental music from text prompts, images, and voice inputs.

## Project Architecture
- **Language**: Python 3.11
- **Framework**: FastAPI + Uvicorn
- **ML Model**: facebook/musicgen-small (loaded lazily on first generation request)
- **External APIs**: OpenAI (for image description and voice transcription/emotion analysis)

## Project Structure
```
backend/
  app.py              - FastAPI application with endpoints
  musicgen_engine.py  - MusicGen model wrapper for audio generation
  openai_vision.py    - OpenAI-based image description for music prompts
  openai_audio.py     - OpenAI-based voice transcription and emotion analysis
  prompt_composer.py  - Combines multi-modal inputs into a single MusicGen prompt
  outputs/            - Generated .wav and .json metadata files
  requirements.txt    - Original dependency list
```

## API Endpoints
- `GET /health` - Health check
- `POST /generate/text` - Generate music from text prompt (JSON body)
- `POST /generate` - Generate music from text/image/voice (multipart form)
- `GET /audio/{filename}` - Download generated .wav file
- `GET /meta/{filename}` - Download generation metadata JSON
- `GET /docs` - Swagger UI API documentation

## Running
The app runs via uvicorn on port 5000:
```
cd backend && uvicorn app:app --host 0.0.0.0 --port 5000
```

## Recent Changes
- Made MusicGen engine loading lazy (loads on first request instead of startup) to allow server to start quickly
- Fixed indentation bug in /generate endpoint where voice processing was outside its conditional block
- Configured for Replit environment (port 5000, 0.0.0.0 host)
