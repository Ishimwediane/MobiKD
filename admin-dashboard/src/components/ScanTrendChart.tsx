'use client';

import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { Download } from 'lucide-react';

interface TrendPoint {
  month: string;
  scans: number;
  healthy: number;
  diseased: number;
}

export default function ScanTrendChart({ data }: { data: TrendPoint[] }) {
  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">Scan Activity Trend</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
            Monthly scans: healthy vs diseased
          </div>
        </div>
        <button className="topbar-icon-btn" aria-label="Download">
          <Download size={15} />
        </button>
      </div>
      <div className="card-body" style={{ paddingTop: 12 }}>
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="gradHealthy" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#3EB75A" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#3EB75A" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradDiseased" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#e53935" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#e53935" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gradTotal" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#FF8C00" stopOpacity={0.18} />
                <stop offset="95%" stopColor="#FF8C00" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" />
            <XAxis dataKey="month" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: 'white', border: '1px solid rgba(0,0,0,0.08)', borderRadius: 10, fontSize: 13 }}
            />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
            <Area type="monotone" dataKey="scans"    stroke="#FF8C00" strokeWidth={2} fill="url(#gradTotal)"    name="Total Scans" dot={false} />
            <Area type="monotone" dataKey="healthy"  stroke="#3EB75A" strokeWidth={2} fill="url(#gradHealthy)"  name="Healthy"     dot={false} />
            <Area type="monotone" dataKey="diseased" stroke="#e53935" strokeWidth={2} fill="url(#gradDiseased)" name="Diseased"    dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
