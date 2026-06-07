'use client';

import { useEffect, useState } from 'react';
import {
  fetchAdminStats, fetchAdminScans,
  shortMonth, diseaseBadge, diseaseColor,
  type AdminStats, type AdminScan,
} from '@/lib/api';
import { LoadingSpinner } from '@/components/StatusWidgets';
import ScanTrendChart from '@/components/ScanTrendChart';
import DiseaseDonut from '@/components/DiseaseDonut';
import UserGrowthChart from '@/components/UserGrowthChart';
import { Activity, ArrowUpRight, TrendingUp, TrendingDown } from 'lucide-react';
import Link from 'next/link';

const KPI_ICONS: Record<string, string> = {
  users: '', scans: '', diseased: '', healthy: '', accuracy: '',
};

export default function DashboardPage() {
  const [stats, setStats]   = useState<AdminStats | null>(null);
  const [scans, setScans]   = useState<AdminScan[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(false);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(false);
      const [s, sc] = await Promise.all([fetchAdminStats(), fetchAdminScans()]);
      if (!s) { setError(true); setLoading(false); return; }
      setStats(s);
      setScans(sc.slice(0, 5));
      setLoading(false);
    }
    load();
  }, []);

  if (loading) return <LoadingSpinner message="Fetching dashboard data from backend…" />;
  if (error || !stats) return (
    <div style={{ textAlign: 'center', padding: 60 }}>
      <div style={{ fontSize: 48, marginBottom: 16 }}>⚠️</div>
      <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }}>Cannot reach the backend</div>
      <div style={{ fontSize: 14, color: 'var(--text-muted)', marginTop: 8 }}>Make sure the Render service is running at <strong>{process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000'}</strong></div>
    </div>
  );

  const trendData = stats.monthly_scan_trend.map(m => ({
    month: shortMonth(m.month),
    scans: m.total,
    healthy: m.healthy,
    diseased: m.diseased,
  }));

  const userGrowthData = stats.monthly_users.map(m => ({
    month: shortMonth(m.month),
    users: m.users,
  }));

  const diseaseDistData = stats.disease_distribution.map(d => ({
    name: d.label,
    value: d.count,
    color: diseaseColor(d.label),
  }));

  const kpiCards = [
    { label: 'Total Users',    value: stats.total_users,    change: '+' + Math.round(stats.total_users * 0.12),   icon: '👥', bg: 'rgba(109,76,151,0.1)', color: '#6d4c97' },
    { label: 'Total Scans',    value: stats.total_scans,    change: '+' + Math.round(stats.total_scans * 0.08),   icon: '📸', bg: 'rgba(255,140,0,0.12)', color: '#FF8C00' },
    { label: 'Diseases Found', value: stats.diseased_count, change: stats.diseased_count > 0 ? '↑' : '—',         icon: '🍂', bg: 'rgba(229,57,53,0.1)', color: '#e53935' },
    { label: 'Healthy Leaves', value: stats.healthy_count,  change: '+' + Math.round(stats.healthy_count * 0.05), icon: '🌿', bg: 'rgba(62,183,90,0.12)', color: '#3EB75A' },
    { label: 'Stage 2 Conf.',  value: `${stats.avg_stage2_confidence}%`, change: '↑',                            icon: '🎯', bg: 'rgba(0,0,0,0.05)', color: '#888' },
  ];

  return (
    <div>
      {/* KPI Cards */}
      <div className="stat-grid">
        {kpiCards.map((k, i) => (
          <div key={k.label} className="stat-card animate-in" style={{ animationDelay: `${i * 0.07}s`, opacity: 0 }}>
            <div className="stat-icon-wrap" style={{ background: k.bg, fontSize: 20 }}>{k.icon}</div>
            <div className="stat-info">
              <div className="stat-value">{k.value.toLocaleString()}</div>
              <div className="stat-label">{k.label}</div>
              <div className="stat-change up" style={{ color: k.color }}>
                <TrendingUp size={11} /> {k.change}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Row 1 */}
      <div className="charts-grid charts-grid-2" style={{ marginBottom: 20 }}>
        <ScanTrendChart data={trendData} />
        <DiseaseDonut data={diseaseDistData} total={stats.total_scans} />
      </div>

      {/* Charts Row 2 */}
      <div className="charts-grid charts-grid-equal">
        <UserGrowthChart data={userGrowthData} total={stats.total_users} />

        {/* Platform Health */}
        <div className="card">
          <div className="card-header">
            <div className="card-title">Platform Health</div>
            <span className="badge badge-green"><Activity size={11} /> Live</span>
          </div>
          <div className="card-body">
            {[
              { label: 'Active Users',      value: `${stats.total_users}`,           color: '#3EB75A', pct: Math.min((stats.total_users / 20) * 100, 100) },
              { label: 'Stage 1 Avg Conf.', value: `${stats.avg_stage1_confidence}%`, color: '#3EB75A', pct: stats.avg_stage1_confidence },
              { label: 'Stage 2 Avg Conf.', value: `${stats.avg_stage2_confidence}%`, color: '#FF8C00', pct: stats.avg_stage2_confidence },
              { label: 'Healthy Ratio',     value: `${stats.total_scans > 0 ? Math.round((stats.healthy_count / stats.total_scans) * 100) : 0}%`, color: '#6d4c97', pct: stats.total_scans > 0 ? (stats.healthy_count / stats.total_scans) * 100 : 0 },
            ].map(item => (
              <div key={item.label} style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{item.label}</span>
                  <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>{item.value}</span>
                </div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${item.pct}%`, background: item.color }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent Scans */}
      <div className="card" style={{ marginTop: 20 }}>
        <div className="card-header" style={{ paddingBottom: 16 }}>
          <div className="card-title">Recent Scans</div>
          <Link href="/scans" className="btn btn-outline" style={{ fontSize: 13, padding: '6px 12px' }}>
            View all <ArrowUpRight size={13} />
          </Link>
        </div>
        <div style={{ overflowX: 'auto', padding: '0 24px 20px' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Farmer</th>
                <th>Stage 1</th>
                <th>Stage 2 Result</th>
                <th>Confidence</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {scans.map(scan => (
                <tr key={scan.id}>
                  <td>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{scan.user_name ?? 'Unknown'}</div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{scan.user_phone}</div>
                  </td>
                  <td>
                    <span className={`badge ${scan.stage1_label === 'Potato Leaf' ? 'badge-green' : 'badge-amber'}`}>
                      {scan.stage1_label}
                    </span>
                  </td>
                  <td>
                    {scan.stage2_label
                      ? <span className={`badge ${diseaseBadge(scan.stage2_label)}`}>{scan.stage2_label}</span>
                      : <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Skipped</span>}
                  </td>
                  <td style={{ fontWeight: 600, fontSize: 13 }}>
                    {scan.stage2_confidence
                      ? `${(scan.stage2_confidence * 100).toFixed(1)}%`
                      : `${(scan.stage1_confidence * 100).toFixed(1)}%`}
                  </td>
                  <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    {new Date(scan.timestamp).toLocaleDateString('en-GB')}
                  </td>
                </tr>
              ))}
              {scans.length === 0 && (
                <tr><td colSpan={5} style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>No scans yet.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
