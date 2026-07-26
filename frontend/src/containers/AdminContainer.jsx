import AdminPanel from "../AdminPanel.jsx";

function AdminContainer({ canManage, currentUserId, onClose, open, request }) {
  if (!open || !canManage) return null;

  return (
    <AdminPanel
      request={request}
      currentUserId={currentUserId}
      onClose={onClose}
    />
  );
}

export default AdminContainer;
