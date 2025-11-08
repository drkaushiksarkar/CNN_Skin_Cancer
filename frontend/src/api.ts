import type {
  CaseListResponse,
  CaseRecord,
  DashboardResponse,
  IntakeFormState,
  IntakeResponse
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

const buildUrl = (path: string) => `${API_BASE}${path}`;

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const message = await res.text();
    throw new Error(message || "Request failed");
  }
  return (await res.json()) as T;
}

export async function fetchCases(): Promise<CaseListResponse> {
  const res = await fetch(buildUrl("/cases"));
  return handleResponse(res);
}

export async function fetchDashboard(): Promise<DashboardResponse> {
  const res = await fetch(buildUrl("/cases/dashboard"));
  return handleResponse(res);
}

export async function submitCase(form: IntakeFormState): Promise<CaseRecord> {
  if (!form.file) {
    throw new Error("Please attach a dermatoscopic image");
  }
  const formData = new FormData();
  formData.append("patient_id", form.patientId);
  formData.append("patient_age", String(form.patientAge));
  formData.append("sex", form.sex);
  formData.append("lesion_site", form.lesionSite);
  formData.append("priority", form.priority);
  if (form.notes) {
    formData.append("notes", form.notes);
  }
  formData.append("image", form.file);

  const res = await fetch(buildUrl("/cases/intake"), {
    method: "POST",
    body: formData
  });
  const payload = await handleResponse<IntakeResponse>(res);
  return payload.case;
}

export async function updateCaseStatus(
  caseId: string,
  status: CaseRecord["status"]
): Promise<CaseRecord> {
  const res = await fetch(buildUrl(`/cases/${caseId}/status`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status })
  });
  return handleResponse(res);
}

export async function addClinicianNote(
  caseId: string,
  message: string,
  author = "clinician"
): Promise<CaseRecord> {
  const res = await fetch(buildUrl(`/cases/${caseId}/notes`), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ author, message })
  });
  return handleResponse(res);
}
