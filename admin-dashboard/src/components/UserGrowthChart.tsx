'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface GrowthPoint { month: string; users: number; }

export default function UserGrowthChart({ data, total }: { data: GrowthPoint[]; total: number }) {
  return (
    <div className="card">
      <div className="card-header">
        <div>
          <div className="card-title">New User Registrations</div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>Monthly new farmers</div>
        </div>
        <span className="badge badge-green">+{total} total</span>
      </div>
      <div className="card-body" style={{ paddingTop: 12 }}>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data} margin={{ top: 0, right: 5, left: -25, bottom: 0 }} barSize={20}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" vertical={false} />
            <XAxis dataKey="month" tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={{ background: 'white', border: '1px solid rgba(0,0,0,0.08)', borderRadius: 10, fontSize: 13 }} cursor={{ fill: 'rgba(62,183,90,0.05)' }} />
            <Bar dataKey="users" name="New Users" radius={[6, 6, 0, 0]}>
              {data.map((_, i) => (
                <Cell key={i} fill={i === data.length - 1 ? '#FF8C00' : 'rgba(62,183,90,0.65)'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
