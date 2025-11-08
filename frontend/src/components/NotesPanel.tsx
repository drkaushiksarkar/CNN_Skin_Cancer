import { FormEvent, useState } from "react";
import type { CaseRecord } from "../types";

type Props = {
  caseRecord: CaseRecord | null;
  onAddNote: (message: string) => Promise<void> | void;
};

function NotesPanel({ caseRecord, onAddNote }: Props) {
  const [note, setNote] = useState("");
  const [submitting, setSubmitting] = useState(false);

  if (!caseRecord) {
    return (
      <div className="card notes-card empty">
        <p>Notes are available once a case is selected.</p>
      </div>
    );
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!note.trim()) {
      return;
    }
    setSubmitting(true);
    try {
      await onAddNote(note.trim());
      setNote("");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="card notes-card">
      <header>
        <div>
          <p className="muted small">Clinician notes</p>
          <h2>{caseRecord.metadata.patient_id}</h2>
        </div>
        <span className="muted small">
          {caseRecord.clinician_notes.length} entries
        </span>
      </header>

      <div className="notes-list">
        {caseRecord.clinician_notes.length === 0 ? (
          <p className="muted small">No notes yet. Add your first summary.</p>
        ) : (
          caseRecord.clinician_notes.map((entry, idx) => (
            <article key={`${entry.created_at}-${idx}`}>
              <p className="note-meta">
                {entry.author} ·
                {new Date(entry.created_at).toLocaleString(undefined, {
                  hour: "2-digit",
                  minute: "2-digit",
                  month: "short",
                  day: "numeric"
                })}
              </p>
              <p>{entry.message}</p>
            </article>
          ))
        )}
      </div>

      <form className="note-form" onSubmit={handleSubmit}>
        <textarea
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="Add review rationale or follow-up plan"
          rows={3}
        />
        <button type="submit" disabled={submitting}>
          {submitting ? "Syncing…" : "Save note"}
        </button>
      </form>
    </div>
  );
}

export default NotesPanel;
