import type { CaseRecord } from "../types";

type Props = {
  caseRecord: CaseRecord | null;
};

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function prettify(label: string): string {
  return label.replaceAll("_", " ");
}

function PredictionSummary({ caseRecord }: Props) {
  if (!caseRecord) {
    return (
      <div className="card prediction-card empty">
        <p>Upload a lesion to unlock instant AI guidance.</p>
      </div>
    );
  }

  const predictions = caseRecord.predictions.slice(0, 4);
  const patient = caseRecord.metadata;
  const imageUrl = caseRecord.image_url
    ? `${import.meta.env.VITE_API_BASE_URL ?? ""}${caseRecord.image_url}`
    : null;

  return (
    <div className="card prediction-card">
      <header>
        <div>
          <p className="muted small">Risk assessment</p>
          <h2>{prettify(predictions[0]?.label ?? "")}</h2>
        </div>
        <span className={`risk-pill ${caseRecord.risk_level}`}>
          {caseRecord.risk_level.toUpperCase()} RISK · {formatPercent(caseRecord.risk_score)}
        </span>
      </header>

      <section className="probability-list">
        {predictions.map((prediction) => (
          <div key={prediction.label} className="probability-row">
            <div>
              <p>{prettify(prediction.label)}</p>
              <p className="muted small">{formatPercent(prediction.probability)}</p>
            </div>
            <div className="bar">
              <span style={{ width: formatPercent(prediction.probability) }} />
            </div>
          </div>
        ))}
      </section>

      <section className="patient-details">
        <div>
          <p className="label">Patient</p>
          <p>{patient.patient_id}</p>
        </div>
        <div>
          <p className="label">Age / Sex</p>
          <p>
            {patient.patient_age} · {patient.sex.toUpperCase()}
          </p>
        </div>
        <div>
          <p className="label">Lesion site</p>
          <p>{prettify(patient.lesion_site)}</p>
        </div>
        <div>
          <p className="label">Priority</p>
          <p>{patient.priority.toUpperCase()}</p>
        </div>
      </section>

      {imageUrl && (
        <section className="image-preview">
          <img src={imageUrl} alt="Dermatoscopic capture" />
          <p className="muted small">Secure intake snapshot</p>
        </section>
      )}
    </div>
  );
}

export default PredictionSummary;
