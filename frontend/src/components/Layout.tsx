import { Outlet, Link, useNavigate } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";

export function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <Link to="/batches" className="text-xl font-bold text-blue-600">
              BankAI
            </Link>
            <Link
              to="/batches/new"
              className="text-sm text-gray-600 hover:text-gray-900"
            >
              New Batch
            </Link>
          </div>
          <div className="flex items-center gap-4">
            {user && (
              <span className="text-sm text-gray-500">{user.email}</span>
            )}
            <button
              onClick={handleLogout}
              className="text-sm text-gray-500 hover:text-gray-700 cursor-pointer"
            >
              Logout
            </button>
          </div>
        </div>
      </nav>
      <main className="max-w-6xl mx-auto px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
