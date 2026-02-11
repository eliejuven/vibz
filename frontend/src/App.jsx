import { useState, useRef, useCallback, useEffect } from 'react'
import {
  ImagePlus,
  FileAudio,
  Mic,
  MicOff,
  SendHorizontal,
  Download,
  ChevronDown,
  ChevronUp,
  Loader2,
  Music,
  X,
} from 'lucide-react'

const API_BASE = ''

function App() {
  const [modelType, setModelType] = useState('baseline')
  const [duration, setDuration] = useState(30)
  const [textPrompt, setTextPrompt] = useState('')
  const [imageFile, setImageFile] = useState(null)
  const [voiceFile, setVoiceFile] = useState(null)
  const [isRecording, setIsRecording] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [showDebug, setShowDebug] = useState(false)

  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const imageInputRef = useRef(null)
  const voiceInputRef = useRef(null)
  const resultRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    if (error) {
      const t = setTimeout(() => setError(''), 5000)
      return () => clearTimeout(t)
    }
  }, [error])

  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [result])

  const autoResize = useCallback(() => {
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 160) + 'px'
    }
  }, [])

  const toggleRecording = useCallback(async () => {
    if (isRecording) {
      mediaRecorderRef.current?.stop()
      setIsRecording(false)
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const file = new File([blob], 'recording.webm', { type: 'audio/webm' })
        setVoiceFile(file)
        stream.getTracks().forEach((t) => t.stop())
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setIsRecording(true)
    } catch {
      setError('Microphone access denied. Please allow microphone access.')
    }
  }, [isRecording])

  const handleSubmit = useCallback(async () => {
    if (!textPrompt.trim() && !imageFile && !voiceFile) {
      setError('Please provide a text prompt, image, or voice input.')
      return
    }

    setIsLoading(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('model_type', modelType)
    formData.append('duration_sec', duration)
    formData.append('temperature', 1.0)
    formData.append('top_k', 250)
    formData.append('text_prompt', textPrompt)

    if (imageFile) formData.append('image', imageFile)
    if (voiceFile) formData.append('voice', voiceFile)

    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data.detail || `Server error (${res.status})`)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Failed to generate music. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }, [textPrompt, imageFile, voiceFile, modelType, duration])

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSubmit()
      }
    },
    [handleSubmit]
  )

  const canSubmit = textPrompt.trim() || imageFile || voiceFile

  return (
    <div className="min-h-screen flex flex-col items-center px-4 py-8 sm:py-12">
      <div className="w-full max-w-[900px]">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-2">
            <Music className="w-8 h-8 text-purple-400" />
            <h1 className="text-5xl sm:text-6xl font-bold bg-gradient-to-r from-purple-400 via-violet-400 to-teal-400 bg-clip-text text-transparent">
              Vibz
            </h1>
          </div>
          <p className="text-gray-400 text-lg font-light">Turn moments into music</p>
        </div>

        <div className="glass rounded-2xl p-5 sm:p-6">
          <div className="flex flex-col sm:flex-row gap-4 mb-5">
            <div className="flex-1">
              <label className="block text-xs font-medium text-gray-400 uppercase tracking-wider mb-1.5">
                Model
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
                className="glass-input w-full rounded-lg px-3 py-2.5 text-white text-sm outline-none focus:ring-1 focus:ring-purple-500/50 appearance-none cursor-pointer"
              >
                <option value="baseline" className="bg-gray-900">Baseline</option>
                <option value="finetuned" disabled className="bg-gray-900">Fine-tuned (coming soon)</option>
              </select>
            </div>

            <div className="flex-1">
              <label className="block text-xs font-medium text-gray-400 uppercase tracking-wider mb-1.5">
                Duration — {duration}s
              </label>
              <div className="flex items-center gap-3 pt-1">
                <span className="text-xs text-gray-500">20s</span>
                <input
                  type="range"
                  min={20}
                  max={45}
                  value={duration}
                  onChange={(e) => setDuration(Number(e.target.value))}
                  className="flex-1"
                />
                <span className="text-xs text-gray-500">45s</span>
              </div>
            </div>
          </div>

          {(imageFile || voiceFile) && (
            <div className="flex flex-wrap gap-2 mb-3">
              {imageFile && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-purple-500/20 text-purple-300 border border-purple-500/30">
                  <ImagePlus className="w-3 h-3" />
                  {imageFile.name}
                  <button onClick={() => setImageFile(null)} className="hover:text-white transition-colors">
                    <X className="w-3 h-3" />
                  </button>
                </span>
              )}
              {voiceFile && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-teal-500/20 text-teal-300 border border-teal-500/30">
                  <FileAudio className="w-3 h-3" />
                  {voiceFile.name}
                  <button onClick={() => setVoiceFile(null)} className="hover:text-white transition-colors">
                    <X className="w-3 h-3" />
                  </button>
                </span>
              )}
            </div>
          )}

          <div className="glass-input rounded-xl p-3">
            <textarea
              ref={textareaRef}
              value={textPrompt}
              onChange={(e) => {
                setTextPrompt(e.target.value)
                autoResize()
              }}
              onKeyDown={handleKeyDown}
              placeholder="Describe the music you want to create..."
              rows={1}
              className="w-full bg-transparent text-white placeholder-gray-500 outline-none resize-none text-sm leading-relaxed mb-2"
            />

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1">
                <input
                  ref={imageInputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.[0]) setImageFile(e.target.files[0])
                    e.target.value = ''
                  }}
                />
                <button
                  onClick={() => imageInputRef.current?.click()}
                  className="p-2 rounded-lg text-gray-400 hover:text-purple-400 hover:bg-white/5 transition-all"
                  title="Upload image"
                >
                  <ImagePlus className="w-5 h-5" />
                </button>

                <input
                  ref={voiceInputRef}
                  type="file"
                  accept="audio/*"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files?.[0]) setVoiceFile(e.target.files[0])
                    e.target.value = ''
                  }}
                />
                <button
                  onClick={() => voiceInputRef.current?.click()}
                  className="p-2 rounded-lg text-gray-400 hover:text-teal-400 hover:bg-white/5 transition-all"
                  title="Upload voice file"
                >
                  <FileAudio className="w-5 h-5" />
                </button>

                <button
                  onClick={toggleRecording}
                  className={`p-2 rounded-lg transition-all ${
                    isRecording
                      ? 'text-red-400 bg-red-500/20 recording-pulse'
                      : 'text-gray-400 hover:text-red-400 hover:bg-white/5'
                  }`}
                  title={isRecording ? 'Stop recording' : 'Record voice'}
                >
                  {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
                </button>
              </div>

              <button
                onClick={handleSubmit}
                disabled={isLoading || !canSubmit}
                className={`w-10 h-10 rounded-full flex items-center justify-center transition-all ${
                  canSubmit && !isLoading
                    ? 'bg-gradient-to-r from-purple-500 to-teal-500 text-white hover:shadow-lg hover:shadow-purple-500/25 hover:scale-105'
                    : 'bg-white/10 text-gray-600 cursor-not-allowed'
                }`}
                title="Generate music"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 spin-slow" />
                ) : (
                  <SendHorizontal className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>

          {isRecording && (
            <div className="mt-3 flex items-center gap-2 text-red-400 text-xs">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
              Recording... Click the mic button to stop.
            </div>
          )}
        </div>

        {error && (
          <div className="mt-4 glass rounded-xl p-4 border-red-500/30 border text-red-300 text-sm flex items-center gap-2">
            <span className="shrink-0">⚠️</span>
            {error}
          </div>
        )}

        {isLoading && (
          <div className="mt-8 flex flex-col items-center gap-3 text-gray-400">
            <Loader2 className="w-10 h-10 spin-slow text-purple-400" />
            <p className="text-sm font-medium">Generating your music...</p>
            <p className="text-xs text-gray-500">This may take a moment</p>
          </div>
        )}

        {result && (
          <div ref={resultRef} className="mt-6 glass rounded-2xl p-5 sm:p-6">
            <div className="flex items-center gap-2 mb-4">
              <Music className="w-5 h-5 text-teal-400" />
              <h2 className="text-lg font-semibold text-white">Your Music</h2>
            </div>

            <audio
              controls
              src={`${API_BASE}${result.download_url}`}
              className="w-full mb-4"
            />

            <div className="flex flex-col sm:flex-row gap-3">
              <a
                href={`${API_BASE}${result.download_url}`}
                download
                className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-purple-500 to-teal-500 text-white text-sm font-medium hover:shadow-lg hover:shadow-purple-500/25 transition-all"
              >
                <Download className="w-4 h-4" />
                Download WAV
              </a>

              <button
                onClick={() => setShowDebug(!showDebug)}
                className="inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl glass-input text-gray-400 text-sm hover:text-white transition-all"
              >
                {showDebug ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                Debug Info
              </button>
            </div>

            {showDebug && (
              <div className="mt-4 glass-input rounded-xl p-4 text-xs text-gray-400 space-y-2 overflow-auto">
                <div>
                  <span className="text-gray-500 font-medium">Audio ID:</span>{' '}
                  <span className="text-gray-300">{result.audio_id}</span>
                </div>
                <div>
                  <span className="text-gray-500 font-medium">Sample Rate:</span>{' '}
                  <span className="text-gray-300">{result.sample_rate} Hz</span>
                </div>
                <div>
                  <span className="text-gray-500 font-medium">Meta URL:</span>{' '}
                  <a
                    href={`${API_BASE}${result.meta_url}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-purple-400 hover:underline break-all"
                  >
                    {result.meta_url}
                  </a>
                </div>
                <div>
                  <span className="text-gray-500 font-medium">Used Prompt:</span>
                  <p className="mt-1 text-gray-300 whitespace-pre-wrap leading-relaxed">{result.used_prompt}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
