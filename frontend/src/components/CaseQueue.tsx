import type { CaseRecord } from "../types";

const statusOptions: { value: CaseRecord["status"]; label: string }[] = [
  { value: "intake", label: "Intake" },
  { value: "review", label: "Review" },
  { value: "escalated", label: "Escalated" },
  { value: "resolved", label: "Resolved" }
];

type Props = {
  cases: CaseRecord[];
  selectedId: string | null;
  onSelect: (caseId: string) => void;
  onStatusChange: (
    caseId: string,
    status: CaseRecord["status"]
  ) => void;
};

function formatDate(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    month: "short",
    day: "numeric"
  });
}

function prettify(text: string): string {
  return text.replaceAll("_", " ");
}

function CaseQueue({ cases, selectedId, onSelect, onStatusChange }: Props) {
  if (!cases.length) {
    return (
      <div className="card queue-card empty">
        <p>No cases yet. Intake a dermatoscopic image to populate the queue.</p>
      </div>
    );
  }

  return (
    <div className="card queue-card">
      <header>
        <div>
          <p className="muted small">Active queue</p>
          <h2>Clinical workbench</h2>
        </div>
        <span className="muted small">{cases.length} tracked cases</span>
      </header>
      <div className="queue-list">
        {cases.map((item) => {
          const selected = item.id === selectedId;
          return (
            <article
              key={item.id}
              className={`queue-item ${selected ? "active" : ""}`}
              onClick={() => onSelect(item.id)}
            >
              <div className="queue-head">
                <div>
                  <h3>{item.metadata.patient_id}</h3>
                  <p className="muted small">
                    {prettify(item.metadata.lesion_site)} · {formatDate(item.created_at)}
                  </p>
                </div>
                <span className={`risk-chip ${item.risk_level}`}>
                  {item.risk_level.toUpperCase()}
                </span>
              </div>
              <p className="muted small">
                AI focus: {prettify(item.predictions[0]?.label ?? "n/a")} ·
                confidence {(item.predictions[0]?.probability ?? 0).toFixed(2)}
              </p>
              <div
                className="queue-actions"
                onClick={(event) => event.stopPropagation()}
              >
                <label>
                  Status
                  <select
                    value={item.status}
                    onChange={(event) =>
                      onStatusChange(
                        item.id,
                        event.target.value as CaseRecord["status"]
                      )
                    }
                  >
                    {statusOptions.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

export default CaseQueue;
