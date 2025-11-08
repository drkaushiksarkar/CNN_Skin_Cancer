import type { CaseRecord } from "../types";

type Props = {
  cases: CaseRecord[];
  onSelect: (caseId: string) => void;
};

function RecentCases({ cases, onSelect }: Props) {
  if (!cases.length) {
    return (
      <div className="card recent-card empty">
        <p>Recent cases will stream in as soon as the first prediction completes.</p>
      </div>
    );
  }

  return (
    <div className="card recent-card">
      <header>
        <h2>Recent escalations</h2>
        <p className="muted small">Tap a row to re-open the chart</p>
      </header>
      <div className="table">
        <div className="table-row head">
          <span>Patient</span>
          <span>Top finding</span>
          <span>Risk</span>
          <span>Updated</span>
        </div>
        {cases.map((item) => (
          <button
            key={item.id}
            className="table-row"
            onClick={() => onSelect(item.id)}
          >
            <span>{item.metadata.patient_id}</span>
            <span>{item.predictions[0]?.label.replaceAll("_", " ") ?? "—"}</span>
            <span className={`risk-chip ${item.risk_level}`}>
              {item.risk_level.toUpperCase()}
            </span>
            <span>
              {new Date(item.updated_at).toLocaleTimeString(undefined, {
                hour: "2-digit",
                minute: "2-digit"
              })}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

export default RecentCases;
