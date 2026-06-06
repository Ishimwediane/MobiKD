/**
 * MobiKD Admin Dashboard — API Service
 *
 * All data fetches go through this file.
 * Strategy: hit the real FastAPI backend first.
 * If the backend is down OR returns empty data, fall back to mockData.json.
 */

import fallback from './mockData.json';

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:8000';

const TIMEOUT_MS = 4000;

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

// ─── API fetch functions ──────────────────────────────────────────────────────

/** Fetch all users. Falls back to mock JSON if backend is empty or unreachable. */
export async function fetchAdminUsers(): Promise<AdminUser[]> {
  const data = await getJson<AdminUser[]>('/admin/users');
  if (data && data.length > 0) return data;
  return fallback.users as AdminUser[];
}

/** Fetch all scans. Falls back to mock JSON if backend is empty or unreachable. */
export async function fetchAdminScans(): Promise<AdminScan[]> {
  const data = await getJson<AdminScan[]>('/admin/scans');
  if (data && data.length > 0) return data;
  return fallback.scans as AdminScan[];
}

/** Fetch aggregate stats. Falls back to mock JSON if backend is empty or unreachable. */
export async function fetchAdminStats(): Promise<AdminStats> {
  const data = await getJson<AdminStats>('/admin/stats');
  if (data && data.total_scans > 0) return data;
  return fallback.stats as AdminStats;
}

/** Model metrics always come from mock JSON (not stored in DB). */
export function getModelMetrics(): ModelMetrics {
  return fallback.modelMetrics as ModelMetrics;
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

/** Check if the backend is reachable. */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetchWithTimeout(`${API_BASE}/docs`, { method: 'HEAD' });
    return res.ok;
  } catch {
    return false;
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
