import { api } from "./client";
import type { UserOut } from "../types";

interface LoginResponse {
  access_token: string;
  user: UserOut;
}

export const authApi = {
  login(email: string, password: string): Promise<LoginResponse> {
    return api.post<LoginResponse>("/auth/login", { email, password });
  },

  me(): Promise<UserOut> {
    return api.get<UserOut>("/auth/me");
  },
};
