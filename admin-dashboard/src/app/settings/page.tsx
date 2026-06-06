'use client';

import { useState } from 'react';
import {
  User, Bell, Shield, Database, Server,
  Save, Eye, EyeOff, Leaf, Globe, Moon, Sun
} from 'lucide-react';

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

function ToggleSwitch({ enabled, onChange, label, description }: {
  enabled: boolean; onChange: () => void; label: string; description?: string;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 0', borderBottom: '1px solid rgba(0,0,0,0.05)' }}>
      <div>
        <div style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-primary)' }}>{label}</div>
        {description && <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{description}</div>}
      </div>
      <button
        onClick={onChange}
        style={{
          width: 44, height: 24, borderRadius: 99,
          background: enabled ? 'var(--color-primary)' : '#e2e8f0',
          border: 'none', cursor: 'pointer', position: 'relative', transition: 'background 0.2s ease',
          flexShrink: 0,
        }}
      >
        <span style={{
          position: 'absolute', top: 2, left: enabled ? 22 : 2,
          width: 20, height: 20, borderRadius: '50%',
          background: 'white', transition: 'left 0.2s ease',
          boxShadow: '0 1px 3px rgba(0,0,0,0.15)',
        }} />
      </button>
    </div>
  );
}

function InputField({ label, value, onChange, type = 'text', placeholder }: {
  label: string; value: string; onChange: (v: string) => void;
  type?: string; placeholder?: string;
}) {
  const [show, setShow] = useState(false);
  const isPass = type === 'password';

  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ display: 'block', fontSize: 13, fontWeight: 500, color: 'var(--text-secondary)', marginBottom: 6 }}>
        {label}
      </label>
      <div style={{ position: 'relative' }}>
        <input
          type={isPass && !show ? 'password' : 'text'}
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          style={{
            width: '100%', padding: '10px 14px',
            paddingRight: isPass ? 40 : 14,
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
        {isPass && (
          <button
            type="button"
            onClick={() => setShow(s => !s)}
            style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
          >
            {show ? <EyeOff size={15} /> : <Eye size={15} />}
          </button>
        )}
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const [adminName, setAdminName] = useState('Administrator');
  const [adminEmail, setAdminEmail] = useState('admin@mobikd.rw');
  const [adminPass, setAdminPass] = useState('');
  const [apiUrl, setApiUrl] = useState('http://127.0.0.1:8000');
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  const [notifications, setNotifications] = useState(true);
  const [autoExport, setAutoExport] = useState(false);
  const [maintenanceMode, setMaintenanceMode] = useState(false);
  const [showConfidence, setShowConfidence] = useState(true);
  const [storeImages, setStoreImages] = useState(true);

  return (
    <div>
      <div className="page-header">
        <h1>Settings</h1>
        <p>Manage your admin profile, system configuration, and notification preferences.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, alignItems: 'start' }}>
        {/* Left Column */}
        <div>
          {/* Admin Profile */}
          <SettingsSection title="Admin Profile" icon={<User size={17} color="var(--color-primary)" />}>
            {/* Avatar */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20, padding: '16px', background: '#f8fafc', borderRadius: 12 }}>
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
              <button className="btn btn-outline" style={{ marginLeft: 'auto', fontSize: 12 }}>Change Photo</button>
            </div>

            <InputField label="Full Name" value={adminName} onChange={setAdminName} placeholder="Administrator" />
            <InputField label="Email Address" value={adminEmail} onChange={setAdminEmail} placeholder="admin@mobikd.rw" />
            <InputField label="New Password" value={adminPass} onChange={setAdminPass} type="password" placeholder="Leave blank to keep current" />

            <button className="btn btn-primary" style={{ marginTop: 4 }}>
              <Save size={14} /> Save Profile
            </button>
          </SettingsSection>

          {/* Appearance */}
          <SettingsSection title="Appearance" icon={<Sun size={17} color="var(--color-primary)" />}>
            <div style={{ display: 'flex', gap: 12, marginBottom: 4 }}>
              {(['light', 'dark'] as const).map(t => (
                <button
                  key={t}
                  onClick={() => setTheme(t)}
                  style={{
                    flex: 1, padding: '14px 16px', borderRadius: 12, cursor: 'pointer',
                    border: `2px solid ${theme === t ? 'var(--color-primary)' : 'rgba(0,0,0,0.08)'}`,
                    background: theme === t ? 'var(--color-primary-light)' : '#f8fafc',
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

        {/* Right Column */}
        <div>
          {/* System Config */}
          <SettingsSection title="System Configuration" icon={<Server size={17} color="var(--color-primary)" />}>
            <InputField label="Backend API URL" value={apiUrl} onChange={setApiUrl} placeholder="http://127.0.0.1:8000" />
            <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
              <button className="btn btn-outline" style={{ flex: 1 }}>
                <Globe size={14} /> Test Connection
              </button>
              <button className="btn btn-primary" style={{ flex: 1 }}>
                <Save size={14} /> Save Config
              </button>
            </div>
          </SettingsSection>

          {/* Notifications */}
          <SettingsSection title="Notifications & Alerts" icon={<Bell size={17} color="var(--color-primary)" />}>
            <ToggleSwitch
              enabled={notifications}
              onChange={() => setNotifications(n => !n)}
              label="Email Notifications"
              description="Receive alerts for new disease outbreaks"
            />
            <ToggleSwitch
              enabled={autoExport}
              onChange={() => setAutoExport(n => !n)}
              label="Weekly Auto-Export"
              description="Send weekly scan report to admin email"
            />
          </SettingsSection>

          {/* Security & Data */}
          <SettingsSection title="Security & Data" icon={<Shield size={17} color="var(--color-primary)" />}>
            <ToggleSwitch
              enabled={storeImages}
              onChange={() => setStoreImages(n => !n)}
              label="Store Scan Images"
              description="Save base64 images in database"
            />
            <ToggleSwitch
              enabled={showConfidence}
              onChange={() => setShowConfidence(n => !n)}
              label="Show Confidence Scores"
              description="Display AI confidence % to farmers"
            />
            <ToggleSwitch
              enabled={maintenanceMode}
              onChange={() => setMaintenanceMode(n => !n)}
              label="Maintenance Mode"
              description="Temporarily disable the API for farmers"
            />

            <div style={{ marginTop: 20 }}>
              <button className="btn" style={{ background: 'rgba(244,63,94,0.08)', color: '#f43f5e', border: '1px solid rgba(244,63,94,0.2)', width: '100%', justifyContent: 'center' }}>
                <Database size={14} /> Purge All Scan History
              </button>
            </div>
          </SettingsSection>
        </div>
      </div>
    </div>
  );
}
