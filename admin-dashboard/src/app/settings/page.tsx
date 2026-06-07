'use client';

import { useState, useEffect } from 'react';
import { User, Save, Sun, Moon } from 'lucide-react';

function SettingsSection({ title, icon, children }: {
  title: string; icon: React.ReactNode; children: React.ReactNode;
}) {
  return (
    <div className="card" style={{ marginBottom: 20 }}>
      <div className="card-header" style={{ paddingBottom: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 10,
            background: 'var(--color-primary-light)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            {icon}
          </div>
          <div className="card-title">{title}</div>
        </div>
      </div>
      <div className="card-body">{children}</div>
    </div>
  );
}

function InputField({ label, value, onChange, placeholder }: {
  label: string; value: string; onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ display: 'block', fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)', marginBottom: 6 }}>
        {label}
      </label>
      <input
        type="text"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          width: '100%', padding: '10px 14px',
          border: '1px solid rgba(0,0,0,0.1)',
          borderRadius: 10, fontSize: 14,
          fontFamily: 'inherit', outline: 'none',
          background: 'white', color: 'var(--text-primary)',
          transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
        }}
        onFocus={e => {
          e.target.style.borderColor = 'var(--color-primary)';
          e.target.style.boxShadow = '0 0 0 3px var(--color-primary-light)';
        }}
        onBlur={e => {
          e.target.style.borderColor = 'rgba(0,0,0,0.1)';
          e.target.style.boxShadow = 'none';
        }}
      />
    </div>
  );
}

export default function SettingsPage() {
  const [adminName, setAdminName] = useState('Administrator');
  const [adminEmail, setAdminEmail] = useState('admin@mobikd.rw');
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  // Read initial theme from document.documentElement class on mount
  useEffect(() => {
    const isDark = document.documentElement.classList.contains('dark');
    setTheme(isDark ? 'dark' : 'light');
  }, []);

  function applyTheme(t: 'light' | 'dark') {
    setTheme(t);
    if (t === 'dark') {
      document.documentElement.classList.add('dark');
      localStorage.setItem('mobikd-theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('mobikd-theme', 'light');
    }
  }

  function handleSave(e: React.FormEvent) {
    e.preventDefault();
    alert('Profile saved successfully!');
  }

  return (
    <div>
      <div className="page-header">
        <h1>Settings</h1>
        <p>Manage your admin profile details and interface appearance preferences.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }}>
        {/* Left Column */}
        <form onSubmit={handleSave}>
          <SettingsSection title="Admin Profile" icon={<User size={17} color="var(--color-primary)" />}>
            {/* Avatar */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20, padding: '16px', background: 'rgba(0,0,0,0.02)', borderRadius: 12 }}>
              <div style={{
                width: 56, height: 56, borderRadius: 16,
                background: 'linear-gradient(135deg, #10b981, #6366f1)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 22, fontWeight: 800, color: 'white',
              }}>A</div>
              <div>
                <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)' }}>{adminName || 'Administrator'}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Super Admin · MobiKD System</div>
              </div>
            </div>

            <InputField label="Full Name" value={adminName} onChange={setAdminName} placeholder="Administrator" />
            <InputField label="Email Address" value={adminEmail} onChange={setAdminEmail} placeholder="admin@mobikd.rw" />

            <button type="submit" className="btn btn-primary" style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 6 }}>
              <Save size={14} /> Save Profile
            </button>
          </SettingsSection>
        </form>

        {/* Right Column */}
        <div>
          {/* Appearance */}
          <SettingsSection title="Appearance" icon={<Sun size={17} color="var(--color-primary)" />}>
            <div style={{ display: 'flex', gap: 12, marginBottom: 4 }}>
              {(['light', 'dark'] as const).map(t => (
                <button
                  key={t}
                  type="button"
                  onClick={() => applyTheme(t)}
                  style={{
                    flex: 1, padding: '14px 16px', borderRadius: 12, cursor: 'pointer',
                    border: `2px solid ${theme === t ? 'var(--color-primary)' : 'rgba(0,0,0,0.08)'}`,
                    background: theme === t ? 'var(--color-primary-light)' : 'rgba(0,0,0,0.01)',
                    display: 'flex', alignItems: 'center', gap: 10, fontFamily: 'inherit',
                  }}
                >
                  {t === 'light' ? <Sun size={16} color={theme === t ? 'var(--color-primary)' : '#94a3b8'} /> : <Moon size={16} color={theme === t ? 'var(--color-primary)' : '#94a3b8'} />}
                  <span style={{ fontSize: 14, fontWeight: 600, color: theme === t ? 'var(--color-primary)' : 'var(--text-secondary)' }}>
                    {t.charAt(0).toUpperCase() + t.slice(1)} Mode
                  </span>
                </button>
              ))}
            </div>
          </SettingsSection>
        </div>
      </div>
    </div>
  );
}
