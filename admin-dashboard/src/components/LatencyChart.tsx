'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface LatencyPoint { day: string; stage1: number; stage2: number; total: number; }

export default function LatencyChart({ data }: { data: LatencyPoint[] }) {
  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">Inference Latency</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>Stage 1 + Stage 2 (ms) — last 7 days</div>
        </div>
        <span className="badge badge-purple">avg {Math.round(data.reduce((s, d) => s + d.total, 0) / Math.max(data.length, 1))}ms</span>
      </div>
      <div className="card-body" style={{ paddingTop: 12 }}>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" />
            <XAxis dataKey="day" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} unit="ms" />
            <Tooltip contentStyle={{ background: 'white', border: '1px solid rgba(0,0,0,0.08)', borderRadius: 10, fontSize: 13 }} formatter={(val: any) => [`${val}ms`, '']} />
            <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
            <Line type="monotone" dataKey="stage1" stroke="#3EB75A" strokeWidth={2.5} dot={{ r: 4, fill: '#3EB75A' }} name="Stage 1" />
            <Line type="monotone" dataKey="stage2" stroke="#FF8C00" strokeWidth={2.5} dot={{ r: 4, fill: '#FF8C00' }} name="Stage 2" />
            <Line type="monotone" dataKey="total"  stroke="#6d4c97" strokeWidth={2}   strokeDasharray="5 3" dot={false}              name="Total"   />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
