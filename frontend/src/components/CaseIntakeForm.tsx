import { ChangeEvent, FormEvent, useState } from "react";
import type { IntakeFormState } from "../types";

const initialState: IntakeFormState = {
  patientId: "",
  patientAge: 52,
  sex: "female",
  lesionSite: "trunk",
  priority: "routine",
  notes: "",
  file: null
};

const sexOptions = [
  { label: "Female", value: "female" },
  { label: "Male", value: "male" },
  { label: "Other", value: "other" }
];

const siteOptions = [
  "trunk",
  "face",
  "scalp",
  "upper_extremity",
  "lower_extremity",
  "palms_soles"
];

const priorityOptions: IntakeFormState["priority"][] = [
  "routine",
  "urgent",
  "stat"
];

type Props = {
  onSubmit: (payload: IntakeFormState) => Promise<boolean>;
  loading: boolean;
};

function CaseIntakeForm({ onSubmit, loading }: Props) {
  const [formState, setFormState] = useState<IntakeFormState>(initialState);
  const [localError, setLocalError] = useState<string | null>(null);

  const handleInputChange = (
    event: ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>
  ) => {
    const { name, value } = event.target;
    setFormState((prev) => ({
      ...prev,
      [name]: name === "patientAge" ? Number(value) : value
    }));
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setFormState((prev) => ({ ...prev, file }));
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!formState.file) {
      setLocalError("Attach a dermatoscopic image to continue");
      return;
    }
    setLocalError(null);
    const ok = await onSubmit(formState);
    if (ok) {
      setFormState((prev) => ({ ...initialState, priority: prev.priority }));
    }
  };

  return (
    <form className="intake-form" onSubmit={handleSubmit}>
      <div>
        <h2>Case intake</h2>
        <p className="muted">
          Capture a lesion image, annotate the patient context, and let DermAssist triage
          the risk instantly.
        </p>
      </div>

      <div className="form-grid">
        <label>
          <span>Patient ID</span>
          <input
            name="patientId"
            value={formState.patientId}
            onChange={handleInputChange}
            placeholder="e.g., ISIC-342"
            required
          />
        </label>
        <label>
          <span>Age</span>
          <input
            type="number"
            name="patientAge"
            value={formState.patientAge}
            onChange={handleInputChange}
            min={1}
            max={120}
            required
          />
        </label>
        <label>
          <span>Sex</span>
          <select name="sex" value={formState.sex} onChange={handleInputChange}>
            {sexOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Lesion site</span>
          <select
            name="lesionSite"
            value={formState.lesionSite}
            onChange={handleInputChange}
          >
            {siteOptions.map((site) => (
              <option key={site} value={site}>
                {site.replaceAll("_", " ")}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Priority</span>
          <select
            name="priority"
            value={formState.priority}
            onChange={handleInputChange}
          >
            {priorityOptions.map((option) => (
              <option key={option} value={option}>
                {option.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
      </div>

      <label>
        <span>Clinical notes</span>
        <textarea
          name="notes"
          value={formState.notes}
          onChange={handleInputChange}
          rows={3}
          placeholder="Dermoscopic patterns, bleeding history, etc."
        />
      </label>

      <div className="upload-field">
        <label className="upload-card">
          <input type="file" accept="image/*" onChange={handleFileChange} />
          <div>
            <p className="upload-title">Drop or browse dermatoscopic image</p>
            <p className="muted small">
              JPEG or PNG • automatically resized to 180×180 px
            </p>
            {formState.file && <p className="selected">{formState.file.name}</p>}
          </div>
        </label>
        <button type="submit" disabled={loading}>
          {loading ? "Running inference…" : "Upload & triage"}
        </button>
      </div>

      {localError && <p className="alert inline">{localError}</p>}
    </form>
  );
}

export default CaseIntakeForm;
