export type Classification = 'HUMAN' | 'SYNTHETIC'
export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'

export interface DetectionResult {
  id: string
  filename: string
  classification: Classification
  confidence: number
  riskLevel: RiskLevel
  explanation: string
  recommendation: string
  analyzedAt: string
  isMock: boolean
  model?: string
  spoofProbability?: number
  bonafideProbability?: number
  riskScore?: number
}
