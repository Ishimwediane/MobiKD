'use client';

import { Bell, Search, Settings } from 'lucide-react';

interface TopbarProps {
  collapsed: boolean;
  title: string;
  subtitle?: string;
}

export default function Topbar({ collapsed, title, subtitle }: TopbarProps) {
  const now = new Date();
  const greeting =
    now.getHours() < 12 ? 'Good morning' :
    now.getHours() < 18 ? 'Good afternoon' : 'Good evening';

  return (
    <header className={`topbar ${collapsed ? 'collapsed' : ''}`}>
      {/* Left */}
      <div className="topbar-left">
        <div>
          <div className="topbar-title">{title}</div>
          {subtitle && <div className="topbar-subtitle">{subtitle}</div>}
        </div>
      </div>

      {/* Right */}
      <div className="topbar-right">
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}>
          {greeting}, Admin
        </div>

        <div className="admin-avatar" title="Admin">A</div>
      </div>
    </header>
  );
}
