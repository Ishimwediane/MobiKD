'use client';

import { useState, useRef, useCallback } from 'react';
import { diagnoseLeaf, diseaseColor, diseaseBadge, type DiagnoseResult } from '@/lib/api';
import { Upload, ScanLine, CheckCircle, AlertTriangle, Cpu, Clock, RefreshCw } from 'lucide-react';

type ScanState = 'idle' | 'scanning' | 'done' | 'error';

export default function AdminScanPage() {
  const [scanState, setScanState] = useState<ScanState>('idle');
  const [result, setResult]       = useState<DiagnoseResult | null>(null);
  const [preview, setPreview]     = useState<string | null>(null);
  const [fileName, setFileName]   = useState<string>('');
  const [isDragging, setIsDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file.');
      return;
    }
    setResult(null);
    setFileName(file.name);
    setPreview(URL.createObjectURL(file));
    setScanState('scanning');

    const res = await diagnoseLeaf(file);
    if (res) {
      setResult(res);
      setScanState('done');
    } else {
      setScanState('error');
    }
  }, []);

  const onFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const reset = () => {
    setScanState('idle');
    setResult(null);
    setPreview(null);
    setFileName('');
    if (fileRef.current) fileRef.current.value = '';
  };

  const isHealthy = result?.stage2Label === 'Healthy';
  const mainColor = result ? diseaseColor(result.stage2Label) : '#6d4c97';

  return (
    <div>
      <div className="page-header">
        <h1>Admin Leaf Scanner</h1>
        <p>Upload a potato leaf image and run the 2-stage AI diagnosis pipeline directly from the dashboard.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, alignItems: 'start' }}>

        {/* ── Left: Upload area ─────────────────────────────── */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)' }}>
            <div className="card-title">Upload Leaf Image</div>
          </div>

          {/* Drop zone */}
          <div
            onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
            style={{
              margin: 24,
              border: `2px dashed ${isDragging ? '#3EB75A' : 'var(--border)'}`,
              borderRadius: 16,
              padding: 32,
              textAlign: 'center',
              cursor: 'pointer',
              background: isDragging ? 'rgba(62,183,90,0.06)' : 'var(--bg-secondary)',
              transition: 'all 0.2s ease',
              minHeight: 200,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 12,
            }}
          >
            <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={onFileChange} id="leaf-upload-input" />
            {preview ? (
              <img
                src={preview}
                alt="Leaf preview"
                style={{ maxHeight: 220, maxWidth: '100%', borderRadius: 12, objectFit: 'cover', boxShadow: '0 4px 20px rgba(0,0,0,0.12)' }}
              />
            ) : (
              <>
                <div style={{ width: 64, height: 64, borderRadius: 16, background: 'rgba(109,76,151,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Upload size={28} color="#6d4c97" />
                </div>
                <div style={{ fontSize: 15, fontWeight: 600, color: 'var(--text-primary)' }}>
                  Drag & drop or click to upload
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
                  Supports JPG, PNG, WEBP
                </div>
              </>
            )}
          </div>

          {/* File info + actions */}
          <div style={{ padding: '0 24px 24px', display: 'flex', flexDirection: 'column', gap: 12 }}>
            {fileName && (
              <div style={{ fontSize: 12, color: 'var(--text-muted)', background: 'var(--bg-secondary)', padding: '8px 12px', borderRadius: 8 }}>
                📎 {fileName}
              </div>
            )}
            <div style={{ display: 'flex', gap: 10 }}>
              <button
                id="scan-button"
                className="btn btn-primary"
                style={{ flex: 1 }}
                onClick={() => fileRef.current?.click()}
                disabled={scanState === 'scanning'}
              >
                <Upload size={15} />
                {preview ? 'Change Image' : 'Choose Image'}
              </button>
              {preview && (
                <button className="btn btn-outline" onClick={reset}>
                  <RefreshCw size={14} /> Reset
                </button>
              )}
            </div>
          </div>
        </div>

        {/* ── Right: Results ────────────────────────────────── */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div style={{ padding: '20px 24px', borderBottom: '1px solid var(--border)' }}>
            <div className="card-title">Diagnosis Result</div>
          </div>

          <div style={{ padding: 24 }}>
            {/* IDLE */}
            {scanState === 'idle' && (
              <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-muted)' }}>
                <ScanLine size={48} style={{ margin: '0 auto 16px', opacity: 0.4 }} />
                <div style={{ fontSize: 15, fontWeight: 600 }}>No scan yet</div>
                <div style={{ fontSize: 13, marginTop: 6 }}>Upload a leaf image to run the AI pipeline.</div>
              </div>
            )}

            {/* SCANNING */}
            {scanState === 'scanning' && (
              <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                <div style={{ fontSize: 42, marginBottom: 16, animation: 'spin 1s linear infinite', display: 'inline-block' }}>🔬</div>
                <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8 }}>Analyzing leaf…</div>
                <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>Running 2-stage AI pipeline on your Render backend. This may take a moment.</div>
                <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
              </div>
            )}

            {/* ERROR */}
            {scanState === 'error' && (
              <div style={{ textAlign: 'center', padding: '40px 20px' }}>
                <AlertTriangle size={48} color="#e53935" style={{ margin: '0 auto 16px' }} />
                <div style={{ fontSize: 16, fontWeight: 700, color: '#e53935', marginBottom: 8 }}>Diagnosis Failed</div>
                <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 20 }}>The backend did not respond. The Render free tier may be waking up. Please try again in 30 seconds.</div>
                <button className="btn btn-outline" onClick={reset}><RefreshCw size={14} /> Try Again</button>
              </div>
            )}

            {/* RESULT */}
            {scanState === 'done' && result && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

                {/* Main result badge */}
                <div style={{ background: `${mainColor}15`, border: `1px solid ${mainColor}40`, borderRadius: 16, padding: 24, textAlign: 'center' }}>
                  <div style={{ fontSize: 40, marginBottom: 8 }}>
                    {isHealthy ? '🌿' : result.stage2Label === 'Early Blight' ? '🟡' : result.stage2Label === 'Late Blight' ? '🔴' : '🟣'}
                  </div>
                  <div style={{ fontSize: 11, fontWeight: 700, color: mainColor, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Final Diagnosis</div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: 'var(--text-primary)' }}>
                    {result.stage2Label ?? result.stage1Label}
                  </div>
                  {result.stage2Confidence && (
                    <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 6 }}>
                      Confidence: <strong style={{ color: mainColor }}>{(result.stage2Confidence * 100).toFixed(1)}%</strong>
                    </div>
                  )}
                </div>

                {/* Stage breakdown */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.8 }}>Stage Breakdown</div>

                  {/* Stage 1 */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-secondary)', padding: '12px 16px', borderRadius: 10 }}>
                    <div>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Stage 1 — Leaf Validator</div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>{result.stage1Label}</div>
                    </div>
                    <span className={`badge ${result.stage1Label === 'Potato Leaf' ? 'badge-green' : 'badge-amber'}`}>
                      {(result.stage1Confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  {/* Stage 2 */}
                  {result.stage2Label && (
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'var(--bg-secondary)', padding: '12px 16px', borderRadius: 10 }}>
                      <div>
                        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Stage 2 — Disease Classifier</div>
                        <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>{result.stage2Label}</div>
                      </div>
                      <span className={`badge ${diseaseBadge(result.stage2Label)}`}>
                        {result.stage2Confidence ? `${(result.stage2Confidence * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                  )}
                </div>

                {/* Meta */}
                <div style={{ display: 'flex', gap: 10 }}>
                  <div style={{ flex: 1, background: 'var(--bg-secondary)', borderRadius: 10, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Clock size={16} color="var(--text-muted)" />
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Latency</div>
                      <div style={{ fontSize: 14, fontWeight: 700 }}>{result.latencyMs.toFixed(0)} ms</div>
                    </div>
                  </div>
                  <div style={{ flex: 1, background: 'var(--bg-secondary)', borderRadius: 10, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 8 }}>
                    <Cpu size={16} color="var(--text-muted)" />
                    <div>
                      <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>Engine</div>
                      <div style={{ fontSize: 14, fontWeight: 700 }}>TFLite CPU</div>
                    </div>
                  </div>
                </div>

                <button className="btn btn-outline" onClick={reset} style={{ width: '100%' }}>
                  <RefreshCw size={14} /> Scan Another Leaf
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
