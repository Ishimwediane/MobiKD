/**
 * MobiKD Admin Dashboard — API Service
 *
 * All live data fetches go directly to the FastAPI backend.
 * Only model performance metrics (not stored in DB) use the static mockData.json.
 */

import mockData from './mockData.json';

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:8000';

const TIMEOUT_MS = 8000;

// ─── Helper: fetch with timeout ───────────────────────────────────────────────

async function fetchWithTimeout(url: string, options?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function getJson<T>(path: string): Promise<T | null> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}${path}`);
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

// ─── Types matching backend responses ────────────────────────────────────────

export interface AdminUser {
  phone: string;
  name: string;
  scan_count: number;
  last_scan: string | null;
}

export interface AdminScan {
  id: string;
  user_phone: string;
  user_name: string | null;
  stage1_label: string;
  stage1_confidence: number;
  stage2_label: string | null;
  stage2_confidence: number | null;
  timestamp: string;
}

export interface DiseaseDist {
  label: string;
  count: number;
}

export interface MonthlyTrend {
  month: string;
  total: number;
  healthy: number;
  diseased: number;
}

export interface MonthlyUsers {
  month: string;
  users: number;
}

export interface AdminStats {
  total_users: number;
  total_scans: number;
  diseased_count: number;
  healthy_count: number;
  disease_distribution: DiseaseDist[];
  monthly_scan_trend: MonthlyTrend[];
  monthly_users: MonthlyUsers[];
  avg_stage1_confidence: number;
  avg_stage2_confidence: number;
}

export interface ModelStage {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  avgLatency: number;
  modelSize: string;
  framework: string;
  inputSize: string;
}

export interface ModelMetrics {
  stage1: ModelStage;
  stage2: ModelStage;
  classes: { label: string; accuracy: number }[];
  latencyHistory: { day: string; stage1: number; stage2: number; total: number }[];
}

export interface DiagnoseResult {
  stage1Label: string;
  stage1Confidence: number;
  stage2Label: string | null;
  stage2Confidence: number | null;
  latencyMs: number;
}

// ─── API fetch functions ──────────────────────────────────────────────────────

/** Fetch all users from the live backend. */
export async function fetchAdminUsers(): Promise<AdminUser[]> {
  const data = await getJson<AdminUser[]>('/admin/users');
  return data ?? [];
}

function normalizeStage1(label: string): string {
  if (label === 'leaf') return 'Potato Leaf';
  if (label === 'not_leaf') return 'Not Leaf';
  return label; // Keep original if already formatted (e.g. Potato Leaf)
}

function normalizeStage2(label: string | null): string | null {
  if (!label) return null;
  if (label === 'healthy') return 'Healthy';
  if (label === 'early_blight') return 'Early Blight';
  if (label === 'late_blight') return 'Late Blight';
  if (label === 'not_potato_leaf') return 'Not Potato Leaf';
  return label;
}

/** Fetch all scans from the live backend. */
export async function fetchAdminScans(): Promise<AdminScan[]> {
  const data = await getJson<AdminScan[]>('/admin/scans');
  if (!data) return [];
  return data.map(scan => ({
    ...scan,
    stage1_label: normalizeStage1(scan.stage1_label),
    stage2_label: normalizeStage2(scan.stage2_label),
  }));
}

/** Fetch aggregate stats from the live backend. */
export async function fetchAdminStats(): Promise<AdminStats | null> {
  const stats = await getJson<AdminStats>('/admin/stats');
  if (!stats) return null;
  return {
    ...stats,
    disease_distribution: (stats.disease_distribution || []).map(d => ({
      label: normalizeStage2(d.label) || 'Unknown',
      count: d.count,
    })),
  };
}

/** Delete a user via the backend. */
export async function deleteUser(phone: string): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/admin/users/${encodeURIComponent(phone)}`, {
      method: 'DELETE',
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** Create a new user from the admin dashboard. */
export async function createUser(phone: string, name: string, password: string): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/admin/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone, name, password }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/** Update an existing user's details from the admin dashboard. */
export async function updateUser(oldPhone: string, phone: string, name: string, password?: string): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/admin/users/${encodeURIComponent(oldPhone)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone, name, password: password || undefined }),
    });
    return res.ok;
  } catch {
    return false;
  }
}


/** Check if the backend is reachable. */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/docs`, { method: 'HEAD' });
    return res.ok;
  } catch {
    return false;
  }
}

/** Model performance metrics are static (not stored in DB) — read from local JSON. */
export function getModelMetrics(): ModelMetrics {
  return mockData.modelMetrics as ModelMetrics;
}

/**
 * Submit a leaf image for AI diagnosis from the admin dashboard.
 * Registers an admin phantom user if not exists, then POSTs to /diagnose.
 */
export async function diagnoseLeaf(imageFile: File): Promise<DiagnoseResult | null> {
  const ADMIN_PHONE = 'admin@mobikd';

  // Ensure admin user exists (safe to call multiple times)
  try {
    await fetch(`${API_BASE}/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone: ADMIN_PHONE, name: 'Admin', password: 'admin_mobikd_2024' }),
    });
  } catch { /* already registered is fine */ }

  try {
    const formData = new FormData();
    formData.append('file', imageFile);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60000); // 60s for cold start
    const res = await fetch(`${API_BASE}/diagnose`, {
      method: 'POST',
      headers: { 'X-User-Phone': ADMIN_PHONE },
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timer);
    if (!res.ok) return null;
    const data = await res.json();
    return {
      stage1Label: data.stage1Label,
      stage1Confidence: data.stage1Confidence,
      stage2Label: data.stage2Label ?? null,
      stage2Confidence: data.stage2Confidence ?? null,
      latencyMs: data.latencyMs,
    };
  } catch {
    return null;
  }
}

// ─── Data transformation helpers ──────────────────────────────────────────────

/** Format a month key like "2026-06" → "Jun" */
export function shortMonth(monthKey: string): string {
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const m = parseInt(monthKey.split('-')[1], 10);
  return months[m - 1] ?? monthKey;
}

/** Get disease bar color by label */
export function diseaseColor(label: string | null): string {
  switch (label) {
    case 'Healthy':        return '#3EB75A';
    case 'Early Blight':   return '#FF8C00';
    case 'Late Blight':    return '#e53935';
    case 'Bacterial Wilt': return '#6d4c97';
    default:               return '#888888';
  }
}

/** Get badge CSS class by disease label */
export function diseaseBadge(label: string | null): string {
  switch (label) {
    case 'Healthy':        return 'badge-green';
    case 'Early Blight':   return 'badge-amber';
    case 'Late Blight':    return 'badge-red';
    case 'Bacterial Wilt': return 'badge-purple';
    default:               return 'badge-slate';
  }
}
