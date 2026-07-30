import { useEffect, useState } from "react";

import { responseErrorMessage, toggleRoleSelection } from "./admin.js";
import { permissionLabel, roleLabel } from "./localization.js";

const emptyForm = {
  username: "",
  email: "",
  password: "",
  roles: ["viewer"],
};

function AdminPanel({ request, currentUserId, onClose }) {
  const [users, setUsers] = useState([]);
  const [roles, setRoles] = useState([]);
  const [form, setForm] = useState(emptyForm);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [savingUserId, setSavingUserId] = useState(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const loadData = async () => {
    setLoading(true);
    setError("");
    try {
      const [usersResponse, rolesResponse] = await Promise.all([
        request("/admin/users"),
        request("/admin/roles"),
      ]);
      if (!usersResponse.ok) {
        throw new Error(await responseErrorMessage(usersResponse, "Kullanıcılar alınamadı."));
      }
      if (!rolesResponse.ok) {
        throw new Error(await responseErrorMessage(rolesResponse, "Roller alınamadı."));
      }
      const [userRows, roleRows] = await Promise.all([
        usersResponse.json(),
        rolesResponse.json(),
      ]);
      setUsers(userRows);
      setRoles(roleRows);
      if (!roleRows.some((role) => role.name === "viewer") && roleRows.length > 0) {
        setForm((current) => ({ ...current, roles: [roleRows[0].name] }));
      }
    } catch (loadError) {
      setError(loadError.message || "Yönetim verileri alınamadı.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const createUser = async (event) => {
    event.preventDefault();
    setCreating(true);
    setError("");
    setNotice("");
    try {
      const response = await request("/admin/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!response.ok) {
        throw new Error(await responseErrorMessage(response, "Kullanıcı oluşturulamadı."));
      }
      const createdUser = await response.json();
      setUsers((current) => [...current, createdUser].sort((a, b) => a.username.localeCompare(b.username)));
      setForm(emptyForm);
      setNotice(`${createdUser.username} oluşturuldu.`);
    } catch (createError) {
      setError(createError.message || "Kullanıcı oluşturulamadı.");
    } finally {
      setCreating(false);
    }
  };

  const updateUser = async (user, changes) => {
    setSavingUserId(user.id);
    setError("");
    setNotice("");
    try {
      const response = await request(`/admin/users/${user.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(changes),
      });
      if (!response.ok) {
        throw new Error(await responseErrorMessage(response, "Kullanıcı güncellenemedi."));
      }
      const updatedUser = await response.json();
      setUsers((current) => current.map((item) => (item.id === user.id ? updatedUser : item)));
      setNotice(`${updatedUser.username} güncellendi.`);
    } catch (updateError) {
      setError(updateError.message || "Kullanıcı güncellenemedi.");
    } finally {
      setSavingUserId(null);
    }
  };

  const changeActiveState = (user) => {
    if (user.is_active && !window.confirm(`${user.username} hesabı kapatılsın mı?`)) return;
    updateUser(user, { is_active: !user.is_active });
  };

  return (
    <section className="mx-auto mb-8 max-w-7xl rounded-lg border border-slate-700 bg-slate-900 p-6 shadow-xl" aria-labelledby="admin-panel-title">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id="admin-panel-title" className="text-xl font-black text-white">Kullanıcı ve Rol Yönetimi</h2>
          <p className="mt-1 text-xs text-slate-500">Rol değişiklikleri kullanıcının mevcut oturumlarını sonlandırır.</p>
        </div>
        <button type="button" onClick={onClose} className="rounded border border-slate-700 px-3 py-1.5 text-sm hover:border-emerald-500">Paneli Kapat</button>
      </div>

      {error && <p role="alert" className="mb-4 rounded border border-red-900 bg-red-950/40 p-3 text-sm text-red-400">{error}</p>}
      {notice && <p role="status" className="mb-4 rounded border border-emerald-900 bg-emerald-950/40 p-3 text-sm text-emerald-400">{notice}</p>}

      <form onSubmit={createUser} className="mb-6 grid gap-3 rounded-lg bg-slate-950 p-4 md:grid-cols-4">
        <input required minLength={3} maxLength={100} pattern="[A-Za-z0-9_.-]+" autoComplete="off" placeholder="Kullanıcı adı" value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm" />
        <input required type="email" autoComplete="off" placeholder="E-posta" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm" />
        <input required type="password" minLength={12} autoComplete="new-password" placeholder="Parola (min. 12)" value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm" />
        <button disabled={creating || loading} className="rounded bg-emerald-500 px-4 py-2 text-sm font-bold text-slate-950 disabled:opacity-50">{creating ? "Oluşturuluyor..." : "Kullanıcı Oluştur"}</button>
        <fieldset className="flex flex-wrap gap-3 md:col-span-4">
          <legend className="mb-2 text-xs text-slate-500">Başlangıç rolleri</legend>
          {roles.map((role) => (
            <label key={role.id} className="flex items-center gap-2 text-sm text-slate-300">
              <input type="checkbox" checked={form.roles.includes(role.name)} onChange={(event) => setForm({ ...form, roles: toggleRoleSelection(form.roles, role.name, event.target.checked) })} />
              {roleLabel(role.name)}
            </label>
          ))}
        </fieldset>
      </form>

      {loading ? (
        <p className="text-sm text-slate-500">Yönetim verileri yükleniyor...</p>
      ) : (
        <div className="space-y-3">
          {users.map((user) => {
            const isCurrentUser = user.id === currentUserId;
            return (
              <article key={user.id} className="grid gap-3 rounded-lg border border-slate-800 bg-slate-950 p-4 md:grid-cols-[1fr_2fr_auto] md:items-center">
                <div>
                  <strong className="block text-sm text-slate-100">{user.username}{isCurrentUser ? " (siz)" : ""}</strong>
                  <span className="text-xs text-slate-500">{user.email}</span>
                </div>
                <div className="flex flex-wrap gap-3">
                  {roles.map((role) => {
                    const checked = user.roles.includes(role.name);
                    return (
                      <label
                        key={role.id}
                        className="flex items-center gap-2 text-sm text-slate-300"
                        title={role.permissions.map(permissionLabel).join(", ")}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={savingUserId === user.id || isCurrentUser || (checked && user.roles.length === 1)}
                          onChange={(event) => updateUser(user, { roles: toggleRoleSelection(user.roles, role.name, event.target.checked) })}
                        />
                        {roleLabel(role.name)}
                      </label>
                    );
                  })}
                </div>
                <button type="button" disabled={savingUserId === user.id || isCurrentUser} onClick={() => changeActiveState(user)} className={`rounded px-3 py-1.5 text-sm font-bold disabled:opacity-40 ${user.is_active ? "border border-red-800 text-red-400" : "border border-emerald-800 text-emerald-400"}`}>
                  {savingUserId === user.id ? "Kaydediliyor..." : user.is_active ? "Hesabı Kapat" : "Aktifleştir"}
                </button>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}

export default AdminPanel;
