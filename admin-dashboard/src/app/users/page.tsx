'use client';

import { useEffect, useState } from 'react';
import { fetchAdminUsers, deleteUser, createUser, updateUser, type AdminUser } from '@/lib/api';
import { LoadingSpinner } from '@/components/StatusWidgets';
import { Search, UserPlus, Users as UsersIcon, UserCheck, UserX, Trash2, Pencil } from 'lucide-react';

export default function UsersPage() {
  const [users, setUsers]   = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch]   = useState('');
  const [deleting, setDeleting] = useState<string | null>(null);

  // Pagination & Filtering
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [sortBy, setSortBy] = useState('name-asc');
  const [activityFilter, setActivityFilter] = useState('all');

  // Modal State
  const [modalOpen, setModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<'add' | 'edit'>('add');
  const [selectedUser, setSelectedUser] = useState<AdminUser | null>(null);
  const [formData, setFormData] = useState({ name: '', phone: '', password: '' });
  const [modalError, setModalError] = useState('');
  const [saving, setSaving] = useState(false);

  async function load() {
    setLoading(true);
    const data = await fetchAdminUsers();
    setUsers(data);
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  // Reset page on filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [search, sortBy, activityFilter, itemsPerPage]);

  const filtered = users
    .filter(u => {
      const matchSearch = u.name.toLowerCase().includes(search.toLowerCase()) || u.phone.includes(search);
      const matchActivity = 
        activityFilter === 'all' ? true :
        activityFilter === 'active' ? u.scan_count > 0 :
        activityFilter === 'inactive' ? u.scan_count === 0 : true;
      return matchSearch && matchActivity;
    })
    .sort((a, b) => {
      if (sortBy === 'name-asc') return a.name.localeCompare(b.name);
      if (sortBy === 'name-desc') return b.name.localeCompare(a.name);
      if (sortBy === 'scans-desc') return b.scan_count - a.scan_count;
      return 0;
    });

  const totalPages = Math.max(1, Math.ceil(filtered.length / itemsPerPage));
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedUsers = filtered.slice(startIndex, endIndex);

  async function handleDelete(phone: string) {
    if (!confirm(`Are you sure you want to delete user ${phone} and all their scans?`)) return;
    setDeleting(phone);
    const ok = await deleteUser(phone);
    if (ok) {
      await load();
    } else {
      alert('Delete failed — backend may be offline or returned an error.');
    }
    setDeleting(null);
  }

  function handleOpenModal(mode: 'add' | 'edit', user?: AdminUser) {
    setModalMode(mode);
    setModalError('');
    if (mode === 'edit' && user) {
      setSelectedUser(user);
      setFormData({ name: user.name, phone: user.phone, password: '' });
    } else {
      setSelectedUser(null);
      setFormData({ name: '', phone: '', password: '' });
    }
    setModalOpen(true);
  }

  function handleCloseModal() {
    setModalOpen(false);
    setFormData({ name: '', phone: '', password: '' });
    setModalError('');
  }

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setModalError('');
    
    const nameClean = formData.name.trim();
    const phoneClean = formData.phone.trim();
    
    if (!nameClean || !phoneClean) {
      setModalError('Name and Phone are required.');
      return;
    }

    setSaving(true);
    try {
      if (modalMode === 'add') {
        if (!formData.password || formData.password.length < 4) {
          setModalError('Password must be at least 4 characters long.');
          setSaving(false);
          return;
        }
        const ok = await createUser(phoneClean, nameClean, formData.password);
        if (ok) {
          await load();
          handleCloseModal();
        } else {
          setModalError('Failed to create user. Phone number might be already registered.');
        }
      } else {
        if (selectedUser) {
          const ok = await updateUser(selectedUser.phone, phoneClean, nameClean, formData.password || undefined);
          if (ok) {
            await load();
            handleCloseModal();
          } else {
            setModalError('Failed to update user. Phone number might be already taken.');
          }
        }
      }
    } catch {
      setModalError('An unexpected network error occurred.');
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <LoadingSpinner message="Fetching users from backend…" />;

  return (
    <div>
      <div className="page-header">
        <h1>User Management</h1>
        <p>All registered farmers using the MobiKD potato disease detection app.</p>
      </div>

      {/* Quick stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 24 }}>
        {[
          { label: 'Total Farmers',  value: users.length,                                   icon: <UsersIcon size={18} color="#6d4c97" />, bg: 'rgba(109,76,151,0.1)' },
          { label: 'With Scans',     value: users.filter(u => u.scan_count > 0).length,     icon: <UserCheck size={18} color="#3EB75A" />, bg: 'rgba(62,183,90,0.12)'  },
          { label: 'No Scans Yet',   value: users.filter(u => u.scan_count === 0).length,   icon: <UserX     size={18} color="#888"    />, bg: 'rgba(0,0,0,0.05)'     },
          { label: 'Total Scans',    value: users.reduce((s, u) => s + u.scan_count, 0),    icon: <UserPlus  size={18} color="#FF8C00" />, bg: 'rgba(255,140,0,0.12)'  },
        ].map(s => (
          <div key={s.label} className="card" style={{ padding: 18, display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, background: s.bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{s.icon}</div>
            <div>
              <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--text-primary)' }}>{s.value}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{s.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Filter bar */}
      <div className="filter-bar" style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
          <div className="filter-search">
            <Search size={15} color="var(--text-muted)" />
            <input placeholder="Search by name or phone…" value={search} onChange={e => setSearch(e.target.value)} />
          </div>

          <select
            className="filter-select"
            value={activityFilter}
            onChange={e => setActivityFilter(e.target.value)}
            style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid rgba(0,0,0,0.1)', background: 'var(--bg-card)', fontSize: '13px', color: 'var(--text-primary)' }}
          >
            <option value="all">All Farmers</option>
            <option value="active">Active (with scans)</option>
            <option value="inactive">Inactive (no scans)</option>
          </select>

          <select
            className="filter-select"
            value={sortBy}
            onChange={e => setSortBy(e.target.value)}
            style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid rgba(0,0,0,0.1)', background: 'var(--bg-card)', fontSize: '13px', color: 'var(--text-primary)' }}
          >
            <option value="name-asc">Sort: Name (A-Z)</option>
            <option value="name-desc">Sort: Name (Z-A)</option>
            <option value="scans-desc">Sort: Scans (High to Low)</option>
          </select>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-secondary)' }}>
            <span>Show:</span>
            <select
              value={itemsPerPage}
              onChange={e => setItemsPerPage(Number(e.target.value))}
              className="filter-select"
              style={{ padding: '6px 10px', borderRadius: '8px', border: '1px solid rgba(0,0,0,0.1)', background: 'var(--bg-card)', fontSize: '13px', color: 'var(--text-primary)' }}
            >
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
            </select>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn btn-outline" onClick={() => handleOpenModal('add')} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <UserPlus size={14} /> Add Farmer
          </button>
          <button className="btn btn-primary" onClick={load}>↻ Refresh</button>
        </div>
      </div>

      {/* Table */}
      <div className="card">
        <div className="card-header" style={{ paddingBottom: 0 }}>
          <div className="card-title">Farmers ({filtered.length})</div>
        </div>
        <div style={{ overflowX: 'auto', padding: '12px 24px 0px' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Farmer</th>
                <th>Phone</th>
                <th>Total Scans</th>
                <th>Last Scan</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {paginatedUsers.map((u, i) => (
                <tr key={u.phone}>
                  <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>{String(startIndex + i + 1).padStart(2, '0')}</td>
                  <td>
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{u.name}</span>
                  </td>
                  <td style={{ fontFamily: 'monospace', fontSize: 13, color: 'var(--text-secondary)' }}>{u.phone}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <div style={{ width: `${Math.min((u.scan_count / 80) * 60, 60)}px`, height: 6, borderRadius: 99, background: 'linear-gradient(90deg,#3EB75A,#FF8C00)' }} />
                      <span style={{ fontWeight: 700, fontSize: 13 }}>{u.scan_count}</span>
                    </div>
                  </td>
                  <td style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    {u.last_scan ? new Date(u.last_scan).toLocaleDateString('en-GB') : '—'}
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button
                        className="btn"
                        style={{ background: 'rgba(109,76,151,0.08)', color: '#6d4c97', padding: '5px 10px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}
                        onClick={() => handleOpenModal('edit', u)}
                      >
                        <Pencil size={12} /> Edit
                      </button>
                      <button
                        className="btn"
                        style={{ background: 'rgba(229,57,53,0.08)', color: '#e53935', padding: '5px 10px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}
                        onClick={() => handleDelete(u.phone)}
                        disabled={deleting === u.phone}
                      >
                        {deleting === u.phone ? '…' : <><Trash2 size={12} /> Delete</>}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {paginatedUsers.length === 0 && (
                <tr><td colSpan={6} style={{ textAlign: 'center', padding: 32, color: 'var(--text-muted)' }}>No farmers found.</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Controls */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '16px 24px',
          borderTop: '1px solid rgba(0,0,0,0.05)',
          background: 'var(--bg-card)',
          borderBottomLeftRadius: '12px',
          borderBottomRightRadius: '12px',
          fontSize: '13px',
          color: 'var(--text-secondary)'
        }}>
          <div>
            Showing <strong>{filtered.length === 0 ? 0 : startIndex + 1}</strong> to <strong>{Math.min(endIndex, filtered.length)}</strong> of <strong>{filtered.length}</strong> entries
          </div>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              className="btn btn-outline"
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              style={{ padding: '6px 12px', fontSize: '12px' }}
            >
              Previous
            </button>

            {Array.from({ length: totalPages }).map((_, idx) => {
              const pageNum = idx + 1;
              if (totalPages > 6 && Math.abs(pageNum - currentPage) > 1 && pageNum !== 1 && pageNum !== totalPages) {
                if (pageNum === 2 || pageNum === totalPages - 1) {
                  return <span key={pageNum} style={{ padding: '6px', color: 'var(--text-muted)' }}>...</span>;
                }
                return null;
              }

              return (
                <button
                  key={pageNum}
                  onClick={() => setCurrentPage(pageNum)}
                  style={{
                    padding: '6px 12px',
                    fontSize: '12px',
                    borderRadius: '6px',
                    cursor: 'pointer',
                    border: pageNum === currentPage ? '1px solid var(--color-primary)' : '1px solid rgba(0,0,0,0.08)',
                    background: pageNum === currentPage ? 'var(--color-primary)' : 'transparent',
                    color: pageNum === currentPage ? 'white' : 'var(--text-secondary)',
                    fontWeight: pageNum === currentPage ? '700' : '500',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {pageNum}
                </button>
              );
            })}

            <button
              className="btn btn-outline"
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              style={{ padding: '6px 12px', fontSize: '12px' }}
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {/* Add/Edit Modal */}
      {modalOpen && (
        <div className="modal-overlay" onClick={handleCloseModal}>
          <div className="modal-container" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{modalMode === 'add' ? 'Add New Farmer' : 'Edit Farmer Profile'}</h3>
              <button className="modal-close" onClick={handleCloseModal}>&times;</button>
            </div>
            <form onSubmit={handleSave}>
              <div className="modal-body">
                {modalError && (
                  <div style={{ background: 'var(--color-danger-light)', color: 'var(--color-danger)', padding: 10, borderRadius: 8, fontSize: 13, marginBottom: 16 }}>
                    {modalError}
                  </div>
                )}
                
                <div className="form-group">
                  <label className="form-label">Farmer Name</label>
                  <input
                    className="form-input"
                    placeholder="e.g. Mugisha Jean"
                    value={formData.name}
                    onChange={e => setFormData({ ...formData, name: e.target.value })}
                    required
                    disabled={saving}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">Phone Number</label>
                  <input
                    className="form-input"
                    placeholder="e.g. 0788123456"
                    value={formData.phone}
                    onChange={e => setFormData({ ...formData, phone: e.target.value })}
                    required
                    disabled={saving}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label">
                    {modalMode === 'add' ? 'Password' : 'New Password (Optional)'}
                  </label>
                  <input
                    type="password"
                    className="form-input"
                    placeholder={modalMode === 'add' ? 'At least 4 characters' : 'Leave empty to keep current'}
                    value={formData.password}
                    onChange={e => setFormData({ ...formData, password: e.target.value })}
                    required={modalMode === 'add'}
                    minLength={4}
                    disabled={saving}
                  />
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-outline" onClick={handleCloseModal} disabled={saving}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={saving}>
                  {saving ? 'Saving…' : modalMode === 'add' ? 'Create User' : 'Save Changes'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
