'use client';

import { useState, useEffect, useRef } from 'react';
import { User, LogOut } from 'lucide-react';
import Link from 'next/link';

interface TopbarProps {
  collapsed: boolean;
  title: string;
  subtitle?: string;
}

export default function Topbar({ collapsed, title, subtitle }: TopbarProps) {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const now = new Date();
  const greeting =
    now.getHours() < 12 ? 'Good morning' :
    now.getHours() < 18 ? 'Good afternoon' : 'Good evening';

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  function handleLogout() {
    if (confirm('Are you sure you want to log out?')) {
      // Direct logout logic or reload
      window.location.href = '/settings';
    }
  }

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

        {/* Profile Dropdown Container */}
        <div style={{ position: 'relative' }} ref={dropdownRef}>
          <button
            className="topbar-icon-btn"
            onClick={() => setDropdownOpen(!dropdownOpen)}
            title="Admin Options"
          >
            <User size={18} />
          </button>

          {dropdownOpen && (
            <div style={{
              position: 'absolute',
              top: '46px',
              right: 0,
              background: 'var(--bg-card, #ffffff)',
              border: '1px solid var(--border-color, rgba(0, 0, 0, 0.08))',
              borderRadius: '10px',
              boxShadow: '0 4px 20px rgba(0, 0, 0, 0.1)',
              padding: '6px',
              minWidth: '160px',
              zIndex: 1000,
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
            }}>
              <Link
                href="/settings"
                onClick={() => setDropdownOpen(false)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 12px',
                  fontSize: '13px',
                  fontWeight: 500,
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  textDecoration: 'none',
                  transition: 'background 0.15s ease',
                }}
                className="dropdown-item"
              >
                <User size={14} /> Profile Settings
              </Link>
              <button
                onClick={handleLogout}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '8px 12px',
                  fontSize: '13px',
                  fontWeight: 500,
                  borderRadius: '6px',
                  color: '#e53935',
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  textAlign: 'left',
                  width: '100%',
                  transition: 'background 0.15s ease',
                }}
                className="dropdown-item-logout"
              >
                <LogOut size={14} /> Log Out
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
