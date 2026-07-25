import { Routes, Route, Navigate } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { Layout } from "./components/Layout";

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
      <Route element={<ProtectedRoute />}>
        <Route element={<Layout />}>
          <Route path="/batches/new" element={<NewBatchPage />} />
          <Route path="/batches/:id" element={<BatchDetailPage />} />
          <Route path="/batches" element={<BatchListPage />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/batches" replace />} />
    </Routes>
  );
}
