import type { DetectionResult, RiskLevel } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const USE_MOCK_API = import.meta.env.VITE_USE_MOCK_API === 'true'
const DETECTION_ENDPOINT = import.meta.env.VITE_DETECTION_ENDPOINT || '/api/v1/predict'

interface PredictionResponse {
  decision: 'SPOOF' | 'BONAFIDE'
  spoof_probability: number
  bonafide_probability: number
  risk_score: number
  model: string
}

const wait = (duration: number) => new Promise((resolve) => setTimeout(resolve, duration))

function mockDetection(file: File): DetectionResult {
  const suspicious = file.name.toLowerCase().match(/clone|synthetic|spoof|fake|ai|deep/) !== null
  const classification = suspicious ? 'SYNTHETIC' : 'HUMAN'
  const confidence = suspicious ? 94 : 97
  const riskLevel: RiskLevel = suspicious ? 'CRITICAL' : 'LOW'

  return {
    id: crypto.randomUUID(),
    filename: file.name,
    classification,
    confidence,
    riskLevel,
    explanation: suspicious
      ? 'The local demo response flags this file for review based on its filename. Connect the FastAPI detection route for model-backed analysis.'
      : 'The local demo response marks this sample as clear. This is a UI-only result and is not a model prediction.',
    recommendation: suspicious
      ? 'Do not rely on the voice alone. Request secondary verification before taking action.'
      : 'Continue with normal verification policy. Model-backed analysis should be enabled before production use.',
    analyzedAt: new Date().toISOString(),
    isMock: true,
  }
}

function riskLevelFor(score: number): RiskLevel {
  if (score >= 0.9) return 'CRITICAL'
  if (score >= 0.8) return 'HIGH'
  if (score >= 0.5) return 'MEDIUM'
  return 'LOW'
}

function mapPrediction(file: File, prediction: PredictionResponse): DetectionResult {
  const isSynthetic = prediction.decision === 'SPOOF'
  const confidence = isSynthetic ? prediction.spoof_probability : prediction.bonafide_probability

  return {
    id: crypto.randomUUID(),
    filename: file.name,
    classification: isSynthetic ? 'SYNTHETIC' : 'HUMAN',
    confidence: Math.round(confidence * 100),
    riskLevel: riskLevelFor(prediction.risk_score),
    explanation: isSynthetic
      ? 'The model found a high likelihood of synthetic or spoofed speech.'
      : 'The model found the audio more consistent with bona-fide human speech.',
    recommendation: isSynthetic
      ? 'Do not rely on the voice alone. Request secondary verification before taking action.'
      : 'Continue with normal verification policy while following your standard review process.',
    analyzedAt: new Date().toISOString(),
    isMock: false,
    model: prediction.model,
    spoofProbability: prediction.spoof_probability,
    bonafideProbability: prediction.bonafide_probability,
    riskScore: prediction.risk_score,
  }
}

async function responseError(response: Response): Promise<Error> {
  try {
    const body = (await response.json()) as { detail?: string }
    if (body.detail) return new Error(body.detail)
  } catch {
    // Use the status fallback when the response is not JSON.
  }
  return new Error(`Detection service returned ${response.status}.`)
}

export async function analyzeAudio(file: File): Promise<DetectionResult> {
  if (USE_MOCK_API) {
    await wait(1800)
    return mockDetection(file)
  }

  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE_URL}${DETECTION_ENDPOINT}`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw await responseError(response)
  }

  return mapPrediction(file, (await response.json()) as PredictionResponse)
}

export const apiConfig = {
  baseUrl: API_BASE_URL,
  usesMock: USE_MOCK_API,
}
