import { useEffect, useState } from "react";
import { listDocuments } from "../api/documents";
import type { DocumentOut } from "../types";

interface Props {
  selectedDocs: DocumentOut[];
  onDocsChange: (docs: DocumentOut[]) => void;
}

export default function ExistingDocumentPicker({ selectedDocs, onDocsChange }: Props) {
  const [items, setItems] = useState<DocumentOut[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listDocuments(50, 0)
      .then((res) => setItems(res.items))
      .catch((err: unknown) =>
        setError(err instanceof Error ? err.message : "Failed to list documents"),
      );
  }, []);

  const selectedIds = new Set(selectedDocs.map((d) => d.id));

  const toggle = (doc: DocumentOut) => {
    if (selectedIds.has(doc.id)) {
      onDocsChange(selectedDocs.filter((d) => d.id !== doc.id));
    } else {
      onDocsChange([...selectedDocs, { ...doc, deduplicated: true }]);
    }
  };

  if (error) {
    return <p className="text-sm text-red-600">{error}</p>;
  }
  if (items.length === 0) {
    return <p className="text-sm text-gray-500">No uploaded statements yet.</p>;
  }

  return (
    <ul className="max-h-40 space-y-1 overflow-y-auto rounded-md border border-gray-200 bg-white p-2">
      {items.map((doc) => (
        <li key={doc.id}>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={selectedIds.has(doc.id)}
              onChange={() => toggle(doc)}
            />
            <span className="truncate">{doc.original_filename}</span>
          </label>
        </li>
      ))}
    </ul>
  );
}
