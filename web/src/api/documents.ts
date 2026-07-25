import { api } from "./client";
import type { DocumentOut } from "../types";

interface DocumentListResponse {
  items: DocumentOut[];
  total: number;
}

export function uploadDocument(file: File): Promise<DocumentOut> {
  const formData = new FormData();
  formData.append("file", file);
  return api.post<DocumentOut>("/documents", formData);
}

export function listDocuments(
  limit: number = 50,
  offset: number = 0,
): Promise<DocumentListResponse> {
  return api.get<DocumentListResponse>(
    `/documents?limit=${limit}&offset=${offset}`,
  );
}
