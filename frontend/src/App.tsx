import { useEffect, useState } from "react";
import {
  addClinicianNote,
  fetchCases,
  fetchDashboard,
  submitCase,
  updateCaseStatus
} from "./api";
import type {
  CaseRecord,
  DashboardResponse,
  IntakeFormState
} from "./types";
import CaseIntakeForm from "./components/CaseIntakeForm";
import PredictionSummary from "./components/PredictionSummary";
import CaseQueue from "./components/CaseQueue";
import InsightCards from "./components/InsightCards";
import RecentCases from "./components/RecentCases";
import NotesPanel from "./components/NotesPanel";

const heroCopy = {
  title: "DermAssist Control Tower",
  subtitle:
    "Guide dermatology teams through intake, AI risk scoring, and escalation decisions in seconds."
};

function App() {
  const [cases, setCases] = useState<CaseRecord[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null);
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [intakeLoading, setIntakeLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedCase = cases.find((c) => c.id === selectedCaseId) ?? null;

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const [casePayload, dashboardPayload] = await Promise.all([
          fetchCases(),
          fetchDashboard()
        ]);
        setCases(casePayload.cases);
        setDashboard(dashboardPayload);
        setSelectedCaseId((prev) => prev ?? casePayload.cases[0]?.id ?? null);
      } catch (err) {
        setError((err as Error).message);
      } finally {
        setLoading(false);
      }
    };
    bootstrap();
  }, []);

  const refreshDashboard = async () => {
    try {
      const next = await fetchDashboard();
      setDashboard(next);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleIntakeSubmit = async (payload: IntakeFormState): Promise<boolean> => {
    setIntakeLoading(true);
    setError(null);
    try {
      const newCase = await submitCase(payload);
      setCases((prev) => [newCase, ...prev]);
      setSelectedCaseId(newCase.id);
      await refreshDashboard();
      return true;
    } catch (err) {
      setError((err as Error).message);
      return false;
    } finally {
      setIntakeLoading(false);
    }
  };

  const handleStatusChange = async (
    caseId: string,
    status: CaseRecord["status"]
  ) => {
    try {
      const updated = await updateCaseStatus(caseId, status);
      setCases((prev) =>
        prev.map((c) => (c.id === caseId ? updated : c))
      );
      if (selectedCaseId === caseId) {
        setSelectedCaseId(updated.id);
      }
      setError(null);
      await refreshDashboard();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  const handleNoteSubmit = async (message: string) => {
    if (!selectedCaseId || !message.trim()) {
      return;
    }
    try {
      const updated = await addClinicianNote(selectedCaseId, message.trim());
      setCases((prev) =>
        prev.map((c) => (c.id === updated.id ? updated : c))
      );
      setError(null);
      await refreshDashboard();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  if (loading) {
    return (
      <div className="app-shell loading">
        <p>Booting DermAssist dashboard…</p>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">AI-guided dermatoscopy</p>
          <h1>{heroCopy.title}</h1>
          <p className="subtitle">{heroCopy.subtitle}</p>
        </div>
        <div className="hero-actions">
          <span className="status-dot" />
          <div>
            <p>Backend status</p>
            <p className="status-ok">Online</p>
          </div>
        </div>
      </header>

      {error && <div className="alert">{error}</div>}

      <section className="grid two-col">
        <div className="panel intake-panel">
          <CaseIntakeForm onSubmit={handleIntakeSubmit} loading={intakeLoading} />
          <NotesPanel caseRecord={selectedCase} onAddNote={handleNoteSubmit} />
        </div>
        <div className="panel summary-panel">
          <PredictionSummary caseRecord={selectedCase} />
          <InsightCards dashboard={dashboard} />
        </div>
      </section>

      <section className="panel queue-panel">
        <CaseQueue
          cases={cases}
          selectedId={selectedCaseId}
          onSelect={setSelectedCaseId}
          onStatusChange={handleStatusChange}
        />
      </section>

      <section className="panel recent-panel">
        <RecentCases
          cases={dashboard?.recent_cases ?? cases.slice(0, 5)}
          onSelect={setSelectedCaseId}
        />
      </section>
    </div>
  );
}

export default App;
