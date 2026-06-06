'use client';

import './globals.css';
import { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import Topbar from '@/components/Topbar';
import { usePathname } from 'next/navigation';

const pageMeta: Record<string, { title: string; subtitle: string }> = {
  '/':          { title: 'Hello, Admin! 👋', subtitle: 'Here\'s what\'s happening with MobiKD today.' },
  '/users':     { title: 'User Management', subtitle: 'All registered farmers using the MobiKD app.' },
  '/scans':     { title: 'Scan History', subtitle: 'Every disease detection scan ever performed.' },
  '/diseases':  { title: 'Disease Analytics', subtitle: 'Deep-dive into potato disease patterns.' },
  '/model':     { title: 'AI Model Statistics', subtitle: 'Real-time performance of the 2-stage TFLite pipeline.' },
  '/settings':  { title: 'Settings', subtitle: 'Configure your admin preferences and system options.' },
};

function AdminShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const pathname = usePathname();
  const meta = pageMeta[pathname] ?? { title: 'MobiKD Admin', subtitle: '' };

  return (
    <div className="admin-shell">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(c => !c)} />
      <div className={`main-content ${collapsed ? 'collapsed' : ''}`}>
        <Topbar collapsed={collapsed} title={meta.title} subtitle={meta.subtitle} />
        <main className="page-body">
          {children}
        </main>
      </div>
    </div>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <title>MobiKD Admin Dashboard</title>
        <meta name="description" content="Admin dashboard for MobiKD potato disease detection system" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body>
        <AdminShell>{children}</AdminShell>
      </body>
    </html>
  );
}
