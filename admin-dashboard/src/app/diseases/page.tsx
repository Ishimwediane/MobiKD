'use client';

import { useEffect, useState } from 'react';
import { fetchAdminStats, shortMonth, diseaseColor, type AdminStats } from '@/lib/api';
import { LoadingSpinner, BackendBanner } from '@/components/StatusWidgets';
import { TrendingUp, TrendingDown, Bug } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
} from 'recharts';

export default function DiseasesPage() {
  const [stats, setStats]   = useState<AdminStats | null>(null);
  const [isMock, setIsMock] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const data = await fetchAdminStats();
      setIsMock(data.total_scans <= 400);
      setStats(data);
      setLoading(false);
    }
    load();
  }, []);

  if (loading) return <LoadingSpinner message="Fetching disease analytics from backend…" />;
  if (!stats) return null;

  const totalDiseased = stats.disease_distribution.filter(d => d.label !== 'Healthy').reduce((s, d) => s + d.count, 0);
  const totalAll = stats.disease_distribution.reduce((s, d) => s + d.count, 0);

  const radarData = stats.disease_distribution.map(d => ({
    subject: d.label.length > 9 ? d.label.split(' ')[0] : d.label,
    value: Math.round((d.count / Math.max(totalAll, 1)) * 100),
  }));

  const trendData = stats.monthly_scan_trend.map(m => ({
    month: shortMonth(m.month),
    healthy: m.healthy,
    diseased: m.diseased,
  }));

  return (
    <div>
      <div className="page-header">
        <h1>Disease Analytics</h1>
        <p>Breakdown of all detected potato diseases with distribution and trends.</p>
      </div>
      <BackendBanner isMock={isMock} />

      {/* Disease summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(stats.disease_distribution.length, 5)}, 1fr)`, gap: 14, marginBottom: 24 }}>
        {stats.disease_distribution.map(d => {
          const color = diseaseColor(d.label);
          const pct = totalAll > 0 ? (d.count / totalAll) * 100 : 0;
          return (
            <div key={d.label} className="card" style={{ padding: 18 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <div style={{ width: 10, height: 10, borderRadius: '50%', background: color }} />
                <span className="badge badge-green" style={{ fontSize: 11 }}><TrendingUp size={10} /> live</span>
              </div>
              <div style={{ fontSize: 26, fontWeight: 800, color: 'var(--text-primary)', letterSpacing: -0.5 }}>
                {d.count.toLocaleString()}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2, fontWeight: 500 }}>{d.label}</div>
              <div style={{ marginTop: 10 }}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>Share: <strong>{pct.toFixed(1)}%</strong></div>
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${pct}%`, background: color }} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Charts row */}
      <div className="charts-grid charts-grid-equal" style={{ marginBottom: 20 }}>
        <div className="card">
          <div className="card-header">
            <div className="card-title">Disease Frequency</div>
            <span className="badge badge-red"><Bug size={11} /> {totalDiseased} cases</span>
          </div>
          <div className="card-body" style={{ paddingTop: 12 }}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={stats.disease_distribution} layout="vertical" margin={{ top: 0, right: 20, left: 10, bottom: 0 }} barSize={22}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="label" tick={{ fontSize: 11, fill: '#64748b' }} axisLine={false} tickLine={false} width={90} />
                <Tooltip contentStyle={{ background: 'white', border: '1px solid rgba(0,0,0,0.08)', borderRadius: 10, fontSize: 13 }} cursor={{ fill: 'rgba(0,0,0,0.03)' }} />
                <Bar dataKey="count" name="Cases" radius={[0, 6, 6, 0]}>
                  {stats.disease_distribution.map((d, i) => <Cell key={i} fill={diseaseColor(d.label)} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><div className="card-title">Distribution Radar</div></div>
          <div className="card-body" style={{ paddingTop: 8 }}>
            <ResponsiveContainer width="100%" height={220}>
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="75%">
                <PolarGrid stroke="rgba(0,0,0,0.08)" />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11, fill: '#64748b' }} />
                <Radar name="Share %" dataKey="value" stroke="#3EB75A" fill="#3EB75A" fillOpacity={0.2} strokeWidth={2} />
                <Tooltip contentStyle={{ background: 'white', border: '1px solid rgba(0,0,0,0.08)', borderRadius: 10, fontSize: 13 }} formatter={(v: number) => [`${v}%`, 'Share']} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Monthly trend */}
      <div className="card">
        <div className="card-header"><div className="card-title">Monthly Disease Detections</div></div>
        <div className="card-body" style={{ paddingTop: 12 }}>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={trendData} margin={{ top: 5, right: 10, left: -20, bottom: 0 }} barSize={16}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" vertical={false} />
              <XAxis dataKey="month" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: 'white', border: '1px solid rgba(0,0,0,0.08)', borderRadius: 10, fontSize: 13 }} />
              <Bar dataKey="healthy"  name="Healthy"  fill="#3EB75A" radius={[4, 4, 0, 0]} />
              <Bar dataKey="diseased" name="Diseased" fill="#e53935" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
