import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import LoginForm from "./LoginForm.jsx";

const props = {
  apiMode: "live",
  credentials: { username: "admin", email: "", password: "invalid-pass" },
  loginError: "Kullanıcı adı veya parola hatalı.",
  loginLoading: false,
  onCredentialsChange: vi.fn(),
  onSubmit: vi.fn(),
  onToggleRegisterMode: vi.fn(),
  registerMode: false,
  registrationEnabled: false,
};

describe("LoginForm", () => {
  it("alanları kalıcı etiketlerle ve hatayı alert olarak sunar", () => {
    render(<LoginForm {...props} />);

    expect(screen.getByLabelText("Kullanıcı adı")).toHaveAttribute(
      "aria-invalid",
      "true",
    );
    expect(screen.getByLabelText("Parola")).toHaveAttribute(
      "aria-describedby",
      "auth-error",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Kullanıcı adı veya parola hatalı.",
    );
  });
});
