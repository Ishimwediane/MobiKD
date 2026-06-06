'use client';

export function LoadingSpinner({ message = 'Loading data…' }: { message?: string }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', minHeight: 200, gap: 16,
    }}>
      <div style={{
        width: 36, height: 36, borderRadius: '50%',
        border: '3px solid rgba(62,183,90,0.15)',
        borderTop: '3px solid #3EB75A',
        animation: 'spin 0.8s linear infinite',
      }} />
      <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>{message}</span>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

export function BackendBanner({ isMock }: { isMock: boolean }) {
  if (!isMock) return null;
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      padding: '10px 16px', marginBottom: 20,
      background: 'rgba(255,140,0,0.08)',
      border: '1px solid rgba(255,140,0,0.25)',
      borderRadius: 10, fontSize: 13,
      color: '#e65100',
    }}>
      <span style={{ fontSize: 16 }}>⚠️</span>
      <span>
        <strong>Backend unreachable or no data yet.</strong> Showing sample data from{' '}
        <code style={{ background: 'rgba(0,0,0,0.06)', padding: '1px 5px', borderRadius: 4 }}>
          mockData.json
        </code>
        . Start the FastAPI server and data will load automatically.
      </span>
    </div>
  );
}

export function ErrorCard({ message }: { message: string }) {
  return (
    <div style={{
      padding: 24, borderRadius: 12,
      background: 'rgba(229,57,53,0.06)',
      border: '1px solid rgba(229,57,53,0.2)',
      color: '#c62828', fontSize: 14,
    }}>
      ❌ {message}
    </div>
  );
}
