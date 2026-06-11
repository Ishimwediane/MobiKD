'use client';

import { useEffect, useState } from 'react';
import { fetchAdminScans, diseaseBadge, type AdminScan } from '@/lib/api';
import { LoadingSpinner } from '@/components/StatusWidgets';
import { Search, Filter, ScanLine, CheckCircle, AlertTriangle, Clock } from 'lucide-react';

export default function ScansPage() {
  const [scans, setScans]   = useState<AdminScan[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch]   = useState('');
  const [filter, setFilter]   = useState('all');

  // Pagination & Filtering
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  const [sortBy, setSortBy] = useState('date-desc');

  useEffect(() => {
    async function load() {
      setLoading(true);
      const data = await fetchAdminScans();
      setScans(data);
      setLoading(false);
    }
    load();
  }, []);

  // Reset page on filter changes
  useEffect(() => {
    setCurrentPage(1);
  }, [search, filter, sortBy, itemsPerPage]);

  const filtered = scans
    .filter(s => {
      const q = search.toLowerCase();
      const matchSearch = (s.user_name ?? '').toLowerCase().includes(q) || s.user_phone.includes(q) || (s.stage2_label ?? '').toLowerCase().includes(q);
      const matchFilter =
        filter === 'all' ? true :
        filter === 'healthy'   ? s.stage2_label === 'Healthy' :
        filter === 'diseased'  ? (s.stage2_label && s.stage2_label !== 'Healthy') :
        filter === 'non-potato'? s.stage1_label === 'Non-Potato' : true;
      return matchSearch && matchFilter;
    })
    .sort((a, b) => {
      if (sortBy === 'date-desc') return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
      if (sortBy === 'date-asc') return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      if (sortBy === 'confidence-desc') return (b.stage2_confidence ?? 0) - (a.stage2_confidence ?? 0);
      return 0;
    });

  const totalPages = Math.max(1, Math.ceil(filtered.length / itemsPerPage));
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedScans = filtered.slice(startIndex, endIndex);

  const totalScans   = scans.length;
  const healthyCount = scans.filter(s => s.stage2_label === 'Healthy').length;
  const diseasedCount = scans.filter(s => s.stage2_label && s.stage2_label !== 'Healthy').length;

  if (loading) return <LoadingSpinner message="Fetching scan history from backend…" />;

  return (
    <div>
      <div className="page-header">
        <h1>Scan History</h1>
        <p>Complete record of all leaf disease detection scans performed via the MobiKD app.</p>
      </div>

      {/* Quick stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 24 }}>
        {[
          { label: 'Total Scans',       value: totalScans,   icon: <ScanLine      size={18} color="#6d4c97" />, bg: 'rgba(109,76,151,0.1)' },
          { label: 'Healthy',           value: healthyCount, icon: <CheckCircle   size={18} color="#3EB75A" />, bg: 'rgba(62,183,90,0.12)'  },
          { label: 'Disease Detected',  value: diseasedCount,icon: <AlertTriangle size={18} color="#e53935" />, bg: 'rgba(229,57,53,0.1)'   },
          { label: 'Filtered Results',  value: filtered.length,icon:<Clock        size={18} color="#FF8C00" />, bg: 'rgba(255,140,0,0.12)'  },
        ].map(s => (
          <div key={s.label} className="card" style={{ padding: 18, display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, background: s.bg, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>{s.icon}</div>
            <div>
              <div style={{ fontSize: 22, fontWeight: 800 }}>{s.value}</div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{s.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="filter-bar" style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
          <div className="filter-search">
            <Search size={15} color="var(--text-muted)" />
            <input placeholder="Search by user, phone, or disease…" value={search} onChange={e => setSearch(e.target.value)} />
          </div>

          <select
            className="filter-select"
            value={filter}
            onChange={e => setFilter(e.target.value)}
            style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid rgba(0,0,0,0.1)', background: 'var(--bg-card)', fontSize: '13px', color: 'var(--text-primary)' }}
          >
            <option value="all">All Results</option>
            <option value="healthy">Healthy Only</option>
            <option value="diseased">Diseases Only</option>
            <option value="non-potato">Non-Potato</option>
          </select>

          <select
            className="filter-select"
            value={sortBy}
            onChange={e => setSortBy(e.target.value)}
            style={{ padding: '8px 12px', borderRadius: '8px', border: '1px solid rgba(0,0,0,0.1)', background: 'var(--bg-card)', fontSize: '13px', color: 'var(--text-primary)' }}
          >
            <option value="date-desc">Newest First</option>
            <option value="date-asc">Oldest First</option>
            <option value="confidence-desc">Highest Confidence</option>
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

        <button className="btn btn-outline"><Filter size={14} /> Export CSV</button>
      </div>

      <div className="card">
        <div className="card-header" style={{ paddingBottom: 0 }}>
          <div className="card-title">All Scans ({filtered.length})</div>
        </div>
        <div style={{ overflowX: 'auto', padding: '12px 24px 0px' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Scan ID</th><th>Farmer</th><th>Stage 1</th><th>Stage 2 Result</th><th>Conf.</th><th>Date</th>
              </tr>
            </thead>
            <tbody>
              {paginatedScans.map(scan => (
                <tr key={scan.id}>
                  <td><span style={{ fontFamily: 'monospace', fontSize: 11, color: 'var(--text-muted)', background: 'rgba(0,0,0,0.04)', padding: '2px 6px', borderRadius: 5 }}>{scan.id.slice(0, 12)}…</span></td>
                  <td>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{scan.user_name ?? 'Unknown'}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{scan.user_phone}</div>
                  </td>
                  <td>
                    <span className={`badge ${scan.stage1_label === 'Potato Leaf' ? 'badge-green' : 'badge-amber'}`}>{scan.stage1_label}</span>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{(scan.stage1_confidence * 100).toFixed(1)}% conf.</div>
                  </td>
                  <td>
                    {scan.stage2_label
                      ? <span className={`badge ${diseaseBadge(scan.stage2_label)}`}>{scan.stage2_label}</span>
                      : <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>—</span>}
                  </td>
                  <td style={{ fontWeight: 700 }}>
                    {scan.stage2_confidence ? `${(scan.stage2_confidence * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    {new Date(scan.timestamp).toLocaleString('en-GB', { dateStyle: 'short', timeStyle: 'short' })}
                  </td>
                </tr>
              ))}
              {paginatedScans.length === 0 && (
                <tr><td colSpan={6} style={{ textAlign: 'center', padding: 32, color: 'var(--text-muted)' }}>No scans match your filter.</td></tr>
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
    </div>
  );
}
