import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import AdminPanel from "./AdminPanel.jsx";

const roles = [
  { id: 1, name: "viewer", permissions: ["history:read"] },
  { id: 2, name: "admin", permissions: ["admin:manage"] },
];
const users = [
  {
    id: "user-1",
    username: "analyst",
    email: "analyst@example.com",
    roles: ["viewer"],
    is_active: true,
  },
];

const jsonResponse = (payload, ok = true) => ({
  ok,
  json: vi.fn().mockResolvedValue(payload),
});

describe("AdminPanel integration", () => {
  it("loads users and roles from both admin endpoints", async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(users))
      .mockResolvedValueOnce(jsonResponse(roles));

    render(
      <AdminPanel request={request} currentUserId="admin-1" onClose={vi.fn()} />,
    );

    expect(await screen.findByText("analyst")).toBeInTheDocument();
    expect(screen.getByText("analyst@example.com")).toBeInTheDocument();
    expect(request).toHaveBeenNthCalledWith(1, "/admin/users");
    expect(request).toHaveBeenNthCalledWith(2, "/admin/roles");
  });

  it("renders an accessible error when admin data cannot be loaded", async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse({ detail: "Yetki yok" }, false))
      .mockResolvedValueOnce(jsonResponse(roles));

    render(
      <AdminPanel request={request} currentUserId="admin-1" onClose={vi.fn()} />,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("Yetki yok");
  });

  it("creates a user and appends it to the visible list", async () => {
    const created = {
      id: "user-2",
      username: "new.user",
      email: "new@example.com",
      roles: ["viewer"],
      is_active: true,
    };
    const request = vi
      .fn()
      .mockResolvedValueOnce(jsonResponse(users))
      .mockResolvedValueOnce(jsonResponse(roles))
      .mockResolvedValueOnce(jsonResponse(created));
    const user = userEvent.setup();
    const { container } = render(
      <AdminPanel request={request} currentUserId="admin-1" onClose={vi.fn()} />,
    );
    await screen.findByText("analyst");

    await user.type(container.querySelector('input[placeholder^="Kullan"]'), "new.user");
    await user.type(container.querySelector('input[type="email"]'), "new@example.com");
    await user.type(container.querySelector('input[type="password"]'), "StrongPassword123!");
    fireEvent.submit(container.querySelector("form"));

    expect(await screen.findByText("new.user")).toBeInTheDocument();
    await waitFor(() => {
      expect(request).toHaveBeenLastCalledWith(
        "/admin/users",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            username: "new.user",
            email: "new@example.com",
            password: "StrongPassword123!",
            roles: ["viewer"],
          }),
        }),
      );
    });
  });
});
