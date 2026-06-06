'use client';

import {
  Users, ScanLine, Bug, Leaf, Brain,
  TrendingUp, TrendingDown
} from 'lucide-react';

const iconMap: Record<string, React.ElementType> = {
  Users, ScanLine, Bug, Leaf, Brain,
};

interface StatCardProps {
  label: string;
  value: string | number;
  change: number;
  icon: string;
  color?: string;
  colorLight?: string;
  delay?: number;
}

const defaultColors: Record<string, { color: string; colorLight: string }> = {
  Users:    { color: '#6d4c97', colorLight: 'rgba(109,76,151,0.1)' },
  ScanLine: { color: '#3EB75A', colorLight: 'rgba(62,183,90,0.12)' },
  Bug:      { color: '#e53935', colorLight: 'rgba(229,57,53,0.1)'  },
  Leaf:     { color: '#3EB75A', colorLight: 'rgba(62,183,90,0.12)' },
  Brain:    { color: '#FF8C00', colorLight: 'rgba(255,140,0,0.12)' },
};

export default function StatCard({ label, value, change, icon, color, colorLight, delay = 0 }: StatCardProps) {
  const Icon = iconMap[icon] || Leaf;
  const colors = defaultColors[icon] ?? { color: '#10b981', colorLight: 'rgba(16,185,129,0.1)' };
  const finalColor = color ?? colors.color;
  const finalColorLight = colorLight ?? colors.colorLight;

  const isUp = change >= 0;

  return (
    <div
      className={`stat-card animate-in`}
      style={{ animationDelay: `${delay * 0.07}s`, opacity: 0 }}
    >
      <div className="stat-icon-wrap" style={{ background: finalColorLight }}>
        <Icon size={22} color={finalColor} strokeWidth={2} />
      </div>
      <div className="stat-info">
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
        <div className={`stat-change ${isUp ? 'up' : 'down'}`}>
          {isUp ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
          {isUp ? '+' : ''}{change}{typeof change === 'number' && Math.abs(change) < 5 ? '%' : ''}
        </div>
      </div>
    </div>
  );
}
