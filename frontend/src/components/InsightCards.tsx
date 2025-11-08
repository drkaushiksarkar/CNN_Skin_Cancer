import type { DashboardResponse } from "../types";

type Props = {
  dashboard: DashboardResponse | null;
};

function InsightCards({ dashboard }: Props) {
  if (!dashboard) {
    return (
      <div className="card insights-card empty">
        <p>Analytics will appear after you process your first case.</p>
      </div>
    );
  }

  const lastUpdated = dashboard.last_updated
    ? new Date(dashboard.last_updated).toLocaleString()
    : "—";

  return (
    <div className="card insights-card">
      <div className="insights-grid">
        <div>
          <p className="label">Total cases</p>
          <h3>{dashboard.total_cases}</h3>
        </div>
        <div>
          <p className="label">High-risk flagged</p>
          <h3>{dashboard.high_risk}</h3>
        </div>
        <div>
          <p className="label">Average risk score</p>
          <h3>{(dashboard.avg_risk * 100).toFixed(1)}%</h3>
        </div>
        <div>
          <p className="label">Last updated</p>
          <h3>{lastUpdated}</h3>
        </div>
      </div>

      <section className="status-breakdown">
        <p className="label">Workflow status</p>
        <div className="chip-row">
          {Object.entries(dashboard.status_breakdown).map(([status, value]) => (
            <span key={status} className="chip">
              {status.toUpperCase()} · {value}
            </span>
          ))}
        </div>
      </section>

      <section>
        <p className="label">Model distribution</p>
        <div className="distribution">
          {dashboard.class_distribution.length === 0 && (
            <p className="muted small">No predictions yet.</p>
          )}
          {dashboard.class_distribution.map((item) => (
            <div key={item.label} className="distribution-row">
              <span>{item.label.replaceAll("_", " ")}</span>
              <div className="bar">
                <span
                  style={{
                    width: `${Math.min(item.count / (dashboard.total_cases || 1), 1) * 100}%`
                  }}
                />
              </div>
              <span>{item.count}</span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default InsightCards;
