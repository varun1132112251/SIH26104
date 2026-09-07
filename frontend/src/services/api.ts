import type { DetectionResult, RiskLevel } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const USE_MOCK_API = import.meta.env.VITE_USE_MOCK_API !== 'false'
const DETECTION_ENDPOINT = import.meta.env.VITE_DETECTION_ENDPOINT

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

export async function analyzeAudio(file: File): Promise<DetectionResult> {
  if (USE_MOCK_API) {
    await wait(1800)
    return mockDetection(file)
  }

  if (!DETECTION_ENDPOINT) {
    throw new Error('No detection endpoint is configured. Set VITE_DETECTION_ENDPOINT when the backend route is ready.')
  }

  const formData = new FormData()
  formData.append('file', file)
  const response = await fetch(`${API_BASE_URL}${DETECTION_ENDPOINT}`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    throw new Error(`Detection service returned ${response.status}.`)
  }

  return (await response.json()) as DetectionResult
}

export const apiConfig = {
  baseUrl: API_BASE_URL,
  usesMock: USE_MOCK_API,
}
