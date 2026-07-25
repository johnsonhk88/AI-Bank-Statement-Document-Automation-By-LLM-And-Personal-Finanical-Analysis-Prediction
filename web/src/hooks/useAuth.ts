import { useEffect } from "react";
import { useAuthStore } from "../stores/authStore";
import { authApi } from "../api/auth";

export function useAuth() {
  const { token, user, login, logout } = useAuthStore();

  useEffect(() => {
    if (token && !user) {
      authApi
        .me()
        .then((u) => {
          useAuthStore.setState({ user: u });
        })
        .catch(() => {
          logout();
        });
    }
  }, [token, user, logout]);

  return {
    isAuthenticated: !!token,
    user,
    login,
    logout,
  };
}
