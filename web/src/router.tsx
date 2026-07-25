import { Routes, Route, Navigate } from "react-router-dom";

function LoginPage() {
  return <div>LoginPage</div>;
}

function BatchListPage() {
  return <div>BatchListPage</div>;
}

function NewBatchPage() {
  return <div>NewBatchPage</div>;
}

function BatchDetailPage() {
  return <div>BatchDetailPage</div>;
}

export function Router() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/batches/new" element={<NewBatchPage />} />
      <Route path="/batches/:id" element={<BatchDetailPage />} />
      <Route path="/batches" element={<BatchListPage />} />
      <Route path="*" element={<Navigate to="/batches" replace />} />
    </Routes>
  );
}
