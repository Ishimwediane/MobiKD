'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import Image from 'next/image';
import {
  LayoutDashboard, Users, ScanLine, Bug, Brain,
  Settings, LogOut, ChevronLeft, ChevronRight, Microscope
} from 'lucide-react';

const navItems = [
  { href: '/',          label: 'Dashboard',         icon: LayoutDashboard },
  { href: '/users',     label: 'Users',              icon: Users           },
  { href: '/scans',     label: 'Scan History',       icon: ScanLine        },
  { href: '/diagnose',  label: 'Admin Scanner',      icon: Microscope      },
  { href: '/diseases',  label: 'Disease Analytics',  icon: Bug             },
  { href: '/model',     label: 'AI Model Stats',     icon: Brain           },
];

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export default function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const pathname = usePathname();

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>

      {/* ── Logo ─────────────────────────────────────────── */}
      <Link href="/" className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <Image
            src="/mobilogo.png"
            alt="MobiKD"
            width={28}
            height={28}
            style={{ objectFit: 'contain' }}
          />
        </div>
        <span className="sidebar-logo-text">MobiKD</span>
      </Link>

      {/* ── Nav ──────────────────────────────────────────── */}
      <nav className="sidebar-nav">
        <div className="sidebar-label">Main Menu</div>

        {navItems.map(({ href, label, icon: Icon }) => {
          const isActive = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`nav-item ${isActive ? 'active' : ''}`}
              title={collapsed ? label : undefined}
            >
              <Icon
                size={18}
                className="nav-icon"
                strokeWidth={isActive ? 2.5 : 2}
              />
              <span className="nav-text">{label}</span>
            </Link>
          );
        })}

        <div className="sidebar-label" style={{ marginTop: 12 }}>System</div>

        <Link
          href="/settings"
          className={`nav-item ${pathname === '/settings' ? 'active' : ''}`}
          title={collapsed ? 'Settings' : undefined}
        >
          <Settings size={18} className="nav-icon" />
          <span className="nav-text">Settings</span>
        </Link>
      </nav>

      {/* ── Footer ───────────────────────────────────────── */}
      <div className="sidebar-footer">
        <button
          className="nav-item"
          style={{ color: '#e53935' }}
          onClick={() => alert('Logout')}
          title={collapsed ? 'Logout' : undefined}
        >
          <LogOut size={18} className="nav-icon" />
          <span className="nav-text">Logout</span>
        </button>

        <button
          onClick={onToggle}
          className="nav-item"
          style={{ marginTop: 4 }}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {collapsed
            ? <ChevronRight size={18} className="nav-icon" />
            : <ChevronLeft size={18} className="nav-icon" />
          }
          <span className="nav-text">Collapse</span>
        </button>
      </div>
    </aside>
  );
}
