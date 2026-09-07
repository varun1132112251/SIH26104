import { ChangeEvent, DragEvent, useEffect, useRef, useState } from 'react'
import { analyzeAudio, apiConfig } from './services/api'
import type { DetectionResult, RiskLevel } from './types'

const ACCEPTED_AUDIO = '.wav,.flac,.ogg,.aiff,.aif'
const SUPPORTED_AUDIO_PATTERN = /\.(wav|flac|ogg|aiff|aif)$/i
const initialHistory: DetectionResult[] = []

const riskMeta: Record<RiskLevel, { label: string; className: string }> = {
  LOW: { label: 'Low risk', className: 'risk-low' },
  MEDIUM: { label: 'Medium risk', className: 'risk-medium' },
  HIGH: { label: 'High risk', className: 'risk-high' },
  CRITICAL: { label: 'Critical risk', className: 'risk-critical' },
}

function formatTime(value: string) {
  return new Intl.DateTimeFormat('en', { hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat('en', { month: 'short', day: 'numeric', year: 'numeric' }).format(new Date(value))
}

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  const [recordedUrl, setRecordedUrl] = useState('')
  const [recordingSeconds, setRecordingSeconds] = useState(0)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisMessage, setAnalysisMessage] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [history, setHistory] = useState<DetectionResult[]>(initialHistory)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      if (recordedUrl) URL.revokeObjectURL(recordedUrl)
      streamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [previewUrl, recordedUrl])

  useEffect(() => {
    if (!isRecording) return
    const timer = window.setInterval(() => setRecordingSeconds((seconds) => seconds + 1), 1000)
    return () => window.clearInterval(timer)
  }, [isRecording])

  const activeFile = selectedFile ?? (recordedBlob ? new File([recordedBlob], 'voice-recording.webm', { type: recordedBlob.type }) : null)

  function clearSelectedAudio() {
    setSelectedFile(null)
    setRecordedBlob(null)
    setError('')
    setResult(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    if (recordedUrl) URL.revokeObjectURL(recordedUrl)
    setPreviewUrl('')
    setRecordedUrl('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  function acceptFile(file: File) {
    if (!SUPPORTED_AUDIO_PATTERN.test(file.name)) {
      setError('Unsupported audio format. Choose WAV, FLAC, OGG, AIFF, or AIF audio.')
      return
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    if (recordedUrl) URL.revokeObjectURL(recordedUrl)
    setSelectedFile(file)
    setRecordedBlob(null)
    setPreviewUrl(URL.createObjectURL(file))
    setRecordedUrl('')
    setResult(null)
    setError('')
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (file) acceptFile(file)
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault()
    setIsDragging(false)
    const file = event.dataTransfer.files[0]
    if (file) acceptFile(file)
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError('This browser does not support audio recording. Upload an audio file instead.')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      chunksRef.current = []
      const recorder = new MediaRecorder(stream)
      recorderRef.current = recorder
      recorder.ondataavailable = (event) => event.data.size > 0 && chunksRef.current.push(event.data)
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        const url = URL.createObjectURL(blob)
        setRecordedBlob(blob)
        setRecordedUrl(url)
        setSelectedFile(null)
        setPreviewUrl('')
        setResult(null)
        setError('')
      }
      recorder.start()
      setRecordingSeconds(0)
      setIsRecording(true)
      setError('')
    } catch {
      setError('Microphone access was unavailable. Check browser permissions or upload an audio file.')
    }
  }

  function stopRecording() {
    recorderRef.current?.stop()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setIsRecording(false)
  }

  async function startAnalysis() {
    if (!activeFile || isAnalyzing) return
    if (!SUPPORTED_AUDIO_PATTERN.test(activeFile.name)) {
      setError('Live recordings are saved as WebM by this browser. Convert the recording to WAV, FLAC, OGG, AIFF, or AIF before analysis.')
      return
    }
    setIsAnalyzing(true)
    setError('')
    setResult(null)
    setAnalysisMessage('Preparing audio signal')
    try {
      await new Promise((resolve) => window.setTimeout(resolve, 550))
      setAnalysisMessage(apiConfig.usesMock ? 'Running local demo analysis' : 'Sending audio to V-SHIELD API')
      const detection = await analyzeAudio(activeFile)
      setResult(detection)
      setHistory((items) => [detection, ...items].slice(0, 6))
      setAnalysisMessage('Analysis complete')
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : 'Analysis failed. Please try again.')
      setAnalysisMessage('Analysis could not be completed')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const displayUrl = selectedFile ? previewUrl : recordedUrl
  const resultMeta = result ? riskMeta[result.riskLevel] : null

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span>V</span></div>
          <div><strong>V-SHIELD</strong><small>SECURITY INTELLIGENCE</small></div>
        </div>
        <div className="sidebar-section-label">Workspace</div>
        <nav className="main-nav" aria-label="Primary navigation">
          <button className="nav-item active" type="button"><span className="nav-icon">⌂</span>Overview</button>
          <button className="nav-item" type="button" onClick={() => document.getElementById('history')?.scrollIntoView({ behavior: 'smooth' })}><span className="nav-icon">◴</span>Analysis history</button>
        </nav>
        <div className="sidebar-footer">
          <div className="system-status"><span className="status-dot" />System ready</div>
          <p>Voice intelligence for high-trust decisions.</p>
          <span className="version-label">M5 FRONTEND / 0.1</span>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div><span className="eyebrow">OPERATIONS CONSOLE</span><h1>Detection overview</h1></div>
          <div className="topbar-meta"><span className="environment-pill"><span className="status-dot" />Local workspace</span><div className="avatar">M5</div></div>
        </header>

        <section className="hero-strip">
          <div><p className="eyebrow accent">AI VOICE DEEPFAKE DETECTION</p><h2>Verify the voice<br /><em>before the decision.</em></h2><p className="hero-copy">Analyze audio signals for synthetic speech and surface the risk context your team needs to act with confidence.</p></div>
          <div className="hero-orbit" aria-hidden="true"><div className="orbit-ring ring-one" /><div className="orbit-ring ring-two" /><div className="orbit-core"><span>V</span></div><div className="orbit-scan" /></div>
          <div className="hero-stat"><span>MODEL STATUS</span><strong>Ready for input</strong><small><i /> {apiConfig.usesMock ? 'DEMO MODE / MOCK RESPONSE' : 'BACKEND API / LIVE RESPONSE'}</small></div>
        </section>

        <div className="content-grid">
          <section className="panel input-panel">
            <div className="panel-heading"><div><span className="section-number">01</span><div><h3>Audio input</h3><p>Upload a sample or record directly.</p></div></div><span className="live-badge">INPUT READY</span></div>
            <div className={`drop-zone ${isDragging ? 'dragging' : ''} ${activeFile ? 'has-file' : ''}`} onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop} onClick={() => !activeFile && fileInputRef.current?.click()} role="button" tabIndex={0} onKeyDown={(event) => event.key === 'Enter' && fileInputRef.current?.click()}>
              <input ref={fileInputRef} type="file" accept={ACCEPTED_AUDIO} onChange={handleFileChange} hidden />
              {activeFile ? <><div className="file-icon">♫</div><div className="file-details"><strong>{activeFile.name}</strong><span>{activeFile.type || 'audio/webm'} <i /> {SUPPORTED_AUDIO_PATTERN.test(activeFile.name) ? 'Ready for analysis' : 'Convert to a supported format before analysis'}</span></div><button className="icon-button" type="button" aria-label="Remove selected audio" onClick={(event) => { event.stopPropagation(); clearSelectedAudio() }}>×</button></> : <><div className="upload-icon">↑</div><div><strong>Drop an audio file here</strong><p>or <u>browse from your device</u></p><small>WAV, FLAC, OGG, AIFF, AIF <i /> Max 30 MB</small></div></>}
            </div>
            {displayUrl && <audio className="audio-player" controls src={displayUrl} />}
            <div className="record-row"><div className={`record-status ${isRecording ? 'recording' : ''}`}><span className="record-dot" />{isRecording ? `Recording 00:${String(recordingSeconds).padStart(2, '0')}` : recordedBlob ? 'Recording captured' : 'Record a live sample'}</div>{isRecording ? <button className="secondary-button stop" type="button" onClick={stopRecording}>Stop recording</button> : <button className="secondary-button" type="button" onClick={startRecording} disabled={isAnalyzing}>Start recording</button>}</div>
            {error && <div className="inline-alert error-alert"><span>!</span>{error}</div>}
            <button className="primary-button analyze-button" type="button" disabled={!activeFile || isAnalyzing || isRecording} onClick={startAnalysis}><span className={isAnalyzing ? 'spinner' : 'spark'}>{isAnalyzing ? '' : '✦'}</span>{isAnalyzing ? 'Analyzing audio...' : 'Start analysis'}<span className="button-arrow">→</span></button>
            {isAnalyzing && <div className="analysis-progress"><div className="progress-track"><span /></div><div><span>{analysisMessage}</span><strong>LIVE</strong></div></div>}
          </section>

          <section className={`panel result-panel ${result ? 'result-ready' : ''}`}>
            <div className="panel-heading"><div><span className="section-number">02</span><div><h3>Detection result</h3><p>Classification and risk assessment.</p></div></div>{result && <span className={`risk-pill ${resultMeta?.className}`}>{result.riskLevel}</span>}</div>
            {result ? <div className={`result-content ${result.classification === 'SYNTHETIC' ? 'suspicious' : 'safe'}`}><div className="result-burst"><div className="result-ring"><span>{result.classification === 'HUMAN' ? '✓' : '!'}</span></div><div><span className="result-kicker">CLASSIFICATION</span><h4>{result.classification}</h4><p>{resultMeta?.label} <i /></p></div></div><div className="confidence-row"><span>Confidence score</span><strong>{result.confidence}%</strong></div><div className="confidence-track"><span style={{ width: `${result.confidence}%` }} /></div><div className="explanation-box"><span className="explanation-icon">{result.classification === 'HUMAN' ? '◈' : '!'}</span><div><strong>{result.classification === 'HUMAN' ? 'Signal appears consistent' : 'Suspicious signal detected'}</strong><p>{result.explanation}</p></div></div><div className="recommendation"><span>RECOMMENDATION</span><p>{result.recommendation}</p></div><small className="demo-note">{result.isMock ? 'Local demo result - not connected to an ML model' : 'Backend analysis result'}</small></div> : <div className="empty-result"><div className="empty-visual"><span>∿</span><i /><i /><i /></div><h4>Awaiting audio signal</h4><p>Select or record an audio sample to begin a detection pass.</p><div className="empty-hint"><span>⌁</span>Results will appear here</div></div>}
          </section>
        </div>

        <section className="panel history-panel" id="history"><div className="panel-heading history-heading"><div><span className="section-number">03</span><div><h3>Recent analyses</h3><p>Your latest detection activity in this workspace.</p></div></div><span className="history-count">{history.length} {history.length === 1 ? 'analysis' : 'analyses'}</span></div>{history.length ? <div className="table-wrap"><table><thead><tr><th>Audio sample</th><th>Classification</th><th>Confidence</th><th>Risk level</th><th>Analyzed</th></tr></thead><tbody>{history.map((item) => <tr key={item.id}><td><span className="table-file-icon">♫</span><strong>{item.filename}</strong></td><td><span className={`table-result ${item.classification === 'SYNTHETIC' ? 'synthetic' : 'human'}`}><i />{item.classification}</span></td><td><strong>{item.confidence}%</strong></td><td><span className={`table-risk ${riskMeta[item.riskLevel].className}`}>{item.riskLevel}</span></td><td><span className="date-stack">{formatTime(item.analyzedAt)}<small>{formatDate(item.analyzedAt)}</small></span></td></tr>)}</tbody></table></div> : <div className="history-empty"><span>◴</span><p>No analyses yet. Your completed detection passes will be listed here.</p></div>}</section>
        <footer className="page-footer"><span>V-SHIELD / AI VOICE DEEPFAKE DETECTION</span><span>Private workspace <i /> {apiConfig.usesMock ? 'MOCK API / DEMO ONLY' : 'Backend API configured'}</span></footer>
      </main>
    </div>
  )
}

export default App
