import { useRef, useState } from "react";
import { uploadDocument } from "../api/documents";
import type { DocumentOut } from "../types";

interface PdfMultiDropzoneProps {
  selectedDocs: DocumentOut[];
  onDocsChange: (docs: DocumentOut[]) => void;
}

export default function PdfMultiDropzone({
  selectedDocs,
  onDocsChange,
}: PdfMultiDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleFiles = async (files: FileList) => {
    setUploading(true);
    setUploadError(null);
    const newDocs: DocumentOut[] = [...selectedDocs];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (!file.name.toLowerCase().endsWith(".pdf")) continue;
      try {
        const doc = await uploadDocument(file);
        newDocs.push(doc);
        onDocsChange([...newDocs]);
      } catch (e: unknown) {
        setUploadError(
          e instanceof Error ? e.message : "Failed to upload document",
        );
      }
    }

    setUploading(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
  };

  const removeDoc = (id: string) => {
    onDocsChange(selectedDocs.filter((d) => d.id !== id));
  };

  return (
    <div>
      <div
        onClick={() => inputRef.current?.click()}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        className="cursor-pointer rounded-lg border-2 border-dashed border-gray-300 p-8 text-center transition-colors hover:border-indigo-400 hover:bg-indigo-50"
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={handleInputChange}
        />
        <p className="text-sm text-gray-600">
          {uploading
            ? "Uploading..."
            : "Drag & drop PDF files here, or click to select files"}
        </p>
        <p className="mt-1 text-xs text-gray-400">PDF files only</p>
      </div>

      {uploadError && (
        <p className="mt-2 text-sm text-red-600">{uploadError}</p>
      )}

      {selectedDocs.length > 0 && (
        <ul className="mt-4 space-y-2">
          {selectedDocs.map((doc) => (
            <li
              key={doc.id}
              className="flex items-center justify-between rounded-md border border-gray-200 bg-white px-3 py-2 text-sm"
            >
              <div className="flex items-center gap-2">
                <span className="truncate">{doc.original_filename}</span>
                {doc.deduplicated && (
                  <span className="rounded bg-amber-100 px-1.5 py-0.5 text-xs font-medium text-amber-800">
                    deduped
                  </span>
                )}
              </div>
              <button
                type="button"
                onClick={() => removeDoc(doc.id)}
                className="ml-2 text-gray-400 hover:text-red-600"
                aria-label={`Remove ${doc.original_filename}`}
              >
                &times;
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
