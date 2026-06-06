'use client';

import { useEffect, useState } from 'react';
import { fetchAdminUsers, deleteUser, type AdminUser } from '@/lib/api';
import { LoadingSpinner, BackendBanner } from '@/components/StatusWidgets';
import { Search, UserPlus, Users as UsersIcon, UserCheck, UserX, Trash2 } from 'lucide-react';

export default function UsersPage() {
  const [users, setUsers]   = useState<AdminUser[]>([]);
  const [isMock, setIsMock] = useState(false);
  const [loading, setLoading] = useState(true);
  const [search, setSearch]   = useState('');
  const [deleting, setDeleting] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    const data = await fetchAdminUsers();
    setIsMock(data.length <= 10 && data.some(u => u.phone === '0788123456'));
    setUsers(data);
    setLoading(false);
  }

  useEffect(() => { load(); }, []);

  const filtered = users.filter(u =>
    u.name.toLowerCase().includes(search.toLowerCase()) ||
    u.phone.includes(search)
  );

  async function handleDelete(phone: string) {
    if (!confirm(`Delete user ${phone} and all their scans?`)) return;
    setDeleting(phone);
    const ok = await deleteUser(phone);
    if (ok) await load();
    else alert('Delete failed — backend may not be running.');
    setDeleting(null);
  }

  if (loading) return <LoadingSpinner message="Fetching users from backend…" />;

  return (
    <div>
      <div className="page-header">
        <h1>User Management</h1>
        <p>All registered farmers using the MobiKD potato disease detection app.</p>
      </div>
      <BackendBanner isMock={isMock} />

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
      <div className="filter-bar">
        <div className="filter-search">
          <Search size={15} color="var(--text-muted)" />
          <input placeholder="Search by name or phone…" value={search} onChange={e => setSearch(e.target.value)} />
        </div>
        <button className="btn btn-primary" onClick={load}>↻ Refresh</button>
      </div>

      {/* Table */}
      <div className="card">
        <div className="card-header" style={{ paddingBottom: 0 }}>
          <div className="card-title">Farmers ({filtered.length})</div>
        </div>
        <div style={{ overflowX: 'auto', padding: '12px 24px 20px' }}>
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
              {filtered.map((u, i) => (
                <tr key={u.phone}>
                  <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>{String(i + 1).padStart(2, '0')}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div style={{ width: 32, height: 32, borderRadius: '50%', background: 'linear-gradient(135deg,#3EB75A,#FF8C00)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', fontWeight: 700, fontSize: 13, flexShrink: 0 }}>
                        {u.name.charAt(0)}
                      </div>
                      <span style={{ fontWeight: 600, fontSize: 13 }}>{u.name}</span>
                    </div>
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
                    <button
                      className="btn"
                      style={{ background: 'rgba(229,57,53,0.08)', color: '#e53935', padding: '5px 10px', fontSize: 12 }}
                      onClick={() => handleDelete(u.phone)}
                      disabled={deleting === u.phone}
                    >
                      {deleting === u.phone ? '…' : <><Trash2 size={12} /> Delete</>}
                    </button>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={6} style={{ textAlign: 'center', padding: 32, color: 'var(--text-muted)' }}>No farmers found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
