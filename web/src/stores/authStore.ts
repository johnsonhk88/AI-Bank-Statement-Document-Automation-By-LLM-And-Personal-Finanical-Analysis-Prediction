import { create } from "zustand";
import type { UserOut } from "../types";

function loadToken(): string | null {
  return localStorage.getItem("token");
}

interface AuthState {
  token: string | null;
  user: UserOut | null;
  login: (token: string, user: UserOut) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  token: loadToken(),
  user: null,

  login: (token: string, user: UserOut) => {
    localStorage.setItem("token", token);
    set({ token, user });
  },

  logout: () => {
    localStorage.removeItem("token");
    set({ token: null, user: null });
  },
}));
