import { Routes, Route, Navigate } from "react-router-dom";
import LoginPage from "./pages/LoginPage";
import NewBatchPage from "./pages/NewBatchPage";
import BatchListPage from "./pages/BatchListPage";
import BatchDetailPage from "./pages/BatchDetailPage";
import CashFlowPage from "./pages/CashFlowPage";
import { ProtectedRoute } from "./components/ProtectedRoute";
import { Layout } from "./components/Layout";

export function Router() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<ProtectedRoute />}>
        <Route element={<Layout />}>
          <Route path="/batches/new" element={<NewBatchPage />} />
          <Route path="/batches/:id" element={<BatchDetailPage />} />
          <Route path="/batches" element={<BatchListPage />} />
          <Route path="/cashflow" element={<CashFlowPage />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/batches" replace />} />
    </Routes>
  );
}
