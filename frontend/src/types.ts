export type Prediction = {
  label: string;
  probability: number;
};

export type CaseMetadata = {
  patient_id: string;
  patient_age: number;
  sex: string;
  lesion_site: string;
  priority: string;
  notes?: string | null;
};

export type ClinicianNote = {
  author: string;
  message: string;
  created_at: string;
};

export type CaseRecord = {
  id: string;
  metadata: CaseMetadata;
  status: "intake" | "review" | "escalated" | "resolved";
  risk_level: "low" | "medium" | "high";
  risk_score: number;
  predictions: Prediction[];
  probability_map: Record<string, number>;
  created_at: string;
  updated_at: string;
  image_url?: string | null;
  clinician_notes: ClinicianNote[];
};

export type CaseListResponse = {
  cases: CaseRecord[];
};

export type DashboardResponse = {
  total_cases: number;
  high_risk: number;
  avg_risk: number;
  last_updated: string | null;
  status_breakdown: Record<string, number>;
  class_distribution: { label: string; count: number }[];
  recent_cases: CaseRecord[];
};

export type IntakeFormState = {
  patientId: string;
  patientAge: number;
  sex: string;
  lesionSite: string;
  priority: "routine" | "urgent" | "stat";
  notes: string;
  file: File | null;
};

export type IntakeResponse = {
  case: CaseRecord;
};
