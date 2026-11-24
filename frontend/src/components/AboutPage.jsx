import React, { useCallback, useEffect, useState } from 'react';
import { Info, Flame, Zap, CheckCircle, Clock, RefreshCw, Satellite, Wrench, Code, Layout, Map, Brain, Image, Radio, Sparkles, Camera, Bell, BarChart2, ClipboardList, BookOpen, AlertCircle } from 'lucide-react';
import { API_BASE } from '../config';

export default function AboutPage() {
  const [status, setStatus] = useState(null);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/status`);
      if (res.ok) setStatus(await res.json());
    } catch (err) {
      console.error('Status fetch failed:', err);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchStatus();
    }, 0);
    return () => clearTimeout(timer);
  }, [fetchStatus]);

  return (
    <div className="about-page">
      <div className="page-header">
        <h1 className="page-title"><Info size={28} style={{marginRight: '8px'}} /> About FireWatch AI</h1>
        <p className="page-subtitle">AI-Driven Forest Fire Detection System</p>
      </div>

      <div className="about-grid">
        {/* System Overview */}
        <div className="about-card hero-card">
          <div className="about-hero-icon"><Flame size={48} /></div>
          <h2>FireWatch AI</h2>
          <p>
            An advanced forest fire detection and monitoring platform that leverages
            NASA satellite data and artificial intelligence to detect, analyze, and
            alert on active fires worldwide in near real-time.
          </p>
        </div>

        {/* System Status */}
        <div className="about-card">
          <h3><Zap size={20} style={{marginRight: '8px', display: 'inline'}} /> System Status</h3>
          {status ? (
            <div className="status-grid">
              <div className="status-item">
                <span className="status-label">Server</span>
                <span className="status-val online">● Running</span>
              </div>
              <div className="status-item">
                <span className="status-label">Active Fires</span>
                <span className="status-val">{status.total_fires}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Last Updated</span>
                <span className="status-val">
                  {status.last_updated
                    ? new Date(status.last_updated).toLocaleString()
                    : 'N/A'}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Analysis</span>
                <span className="status-val">
                  {status.analysis_available ? <><CheckCircle size={14} style={{display:'inline'}} /> Available</> : <><Clock size={14} style={{display:'inline'}} /> Pending</>}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Data Source</span>
                <span className="status-val" style={{ color: status.is_demo_data ? '#f59e0b' : '#38bdf8', fontWeight: 600 }}>
                  {status.is_demo_data ? 'DEMO / SIMULATED' : `NASA FIRMS (${status.satellite_id || status.satellite || 'VIIRS'})`}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">API Status</span>
                <span className="status-val">{status.api_status ? status.api_status.toUpperCase() : 'UNKNOWN'}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Hotspot Breakdown</span>
                <span className="status-val">
                  {status.is_demo_data ? `${status.demo_count ?? status.total_fires} simulated (0 genuine)` : `${status.genuine_count ?? status.total_fires} genuine NASA`}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Last NASA Sync</span>
                <span className="status-val">
                  {status.last_successful_fetch ? new Date(status.last_successful_fetch).toLocaleString() : 'No successful NASA sync'}
                </span>
              </div>
              <div className="status-item">
                <span className="status-label">Fetching</span>
                <span className="status-val">
                  {status.is_fetching ? <><RefreshCw size={14} style={{display:'inline'}} /> In progress</> : <><CheckCircle size={14} style={{display:'inline'}} /> Idle</>}
                </span>
              </div>
            </div>
          ) : (
            <p className="about-muted">Loading status...</p>
          )}
        </div>

        {/* Data Sources */}
        <div className="about-card">
          <h3><Satellite size={20} style={{marginRight: '8px', display: 'inline'}} /> Data Sources</h3>
          <div className="about-list">
            <div className="about-list-item">
              <strong>NASA FIRMS</strong>
              <p>Fire Information for Resource Management System — provides near real-time active fire data from satellite observations.</p>
            </div>
            <div className="about-list-item">
              <strong>VIIRS S-NPP</strong>
              <p>Visible Infrared Imaging Radiometer Suite on Suomi NPP satellite. 375m resolution thermal anomaly detection.</p>
            </div>
            <div className="about-list-item">
              <strong>MODIS</strong>
              <p>Moderate Resolution Imaging Spectroradiometer on Terra/Aqua satellites. 1km resolution fire detection.</p>
            </div>
          </div>
        </div>

        {/* AI Model Audit & Evaluation Status */}
        <div className="about-card">
          <h3><Brain size={20} style={{marginRight: '8px', display: 'inline'}} /> AI Models & Evaluation Audit</h3>
          {status?.models?.models ? (
            <div className="about-list">
              <div className="about-list-item">
                <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                  <strong>EfficientNet-B0 (Binary Classifier)</strong>
                  <span className={`status-val ${status.models.models.efficientnet?.can_detect_fire ? 'online' : ''}`} style={{fontSize:'12px', fontWeight:600}}>
                    {status.models.models.efficientnet?.custom_weights_loaded ? 'CUSTOM WEIGHTS LOADED' : 'IMAGENET FALLBACK'}
                  </span>
                </div>
                <p>{status.models.models.efficientnet?.note}</p>
                <div style={{fontSize:'12px', color:'var(--text-secondary, #94a3b8)', marginTop:'4px'}}>
                  Evaluation: <strong>{status.models.models.efficientnet?.evaluation_status?.toUpperCase()}</strong>
                  {status.models.models.efficientnet?.metrics && (
                    <span> • Test Acc: {(status.models.models.efficientnet.metrics.accuracy * 100).toFixed(1)}% • F1: {(status.models.models.efficientnet.metrics.f1_score * 100).toFixed(1)}%</span>
                  )}
                </div>
              </div>

              <div className="about-list-item">
                <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                  <strong>YOLOv8n (Object Detection)</strong>
                  <span style={{fontSize:'12px', fontWeight:600, color: status.models.models.yolov8?.can_detect_fire ? '#10b981' : '#f59e0b'}}>
                    {status.models.models.yolov8?.custom_weights_loaded ? 'CUSTOM FIRE WEIGHTS' : 'COCO PRETRAINED (FALLBACK)'}
                  </span>
                </div>
                <p>{status.models.models.yolov8?.note}</p>
                <div style={{fontSize:'12px', color:'var(--text-secondary, #94a3b8)', marginTop:'4px'}}>
                  Evaluation: <strong>{status.models.models.yolov8?.evaluation_status?.toUpperCase()}</strong> • Fire Classes: <strong>{status.models.models.yolov8?.can_detect_fire ? 'Present' : 'Absent (COCO 80 classes)'}</strong>
                </div>
              </div>

              <div className="about-list-item">
                <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                  <strong>CNN-LSTM (Temporal Sequence)</strong>
                  <span style={{fontSize:'12px', fontWeight:600, color: status.models.models.cnn_lstm?.can_detect_fire ? '#10b981' : '#ef4444'}}>
                    {status.models.models.cnn_lstm?.custom_weights_loaded ? 'CUSTOM WEIGHTS' : 'RANDOM WEIGHTS (NOT EVALUATED)'}
                  </span>
                </div>
                <p>{status.models.models.cnn_lstm?.note}</p>
                <div style={{fontSize:'12px', color:'var(--text-secondary, #94a3b8)', marginTop:'4px'}}>
                  Evaluation: <strong>{status.models.models.cnn_lstm?.evaluation_status?.toUpperCase()}</strong> • Operational: <strong>{status.models.models.cnn_lstm?.ready ? 'Ready' : 'Excluded from Ensemble'}</strong>
                </div>
              </div>

              <div className="about-list-item">
                <div style={{display:'flex', justifyContent:'space-between', alignItems:'center'}}>
                  <strong>Color & Texture Heuristic</strong>
                  <span style={{fontSize:'12px', fontWeight:600, color: '#38bdf8'}}>
                    RULE-BASED (SEPARATED)
                  </span>
                </div>
                <p>{status.models.heuristic_evaluation?.note}</p>
                {status.models.heuristic_evaluation?.metrics && (
                  <div style={{fontSize:'12px', color:'var(--text-secondary, #94a3b8)', marginTop:'4px'}}>
                    Separately Evaluated • Test Acc: {(status.models.heuristic_evaluation.metrics.accuracy * 100).toFixed(1)}% • F1: {(status.models.heuristic_evaluation.metrics.f1_score * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            </div>
          ) : (
            <p className="about-muted">Loading model audit...</p>
          )}
        </div>

        {/* Tech Stack */}
        <div className="about-card">
          <h3><Wrench size={20} style={{marginRight: '8px', display: 'inline'}} /> Technology Stack</h3>
          <div className="tech-grid">
            <div className="tech-item">
              <span className="tech-icon"><Layout size={20} /></span>
              <span>React + Vite</span>
            </div>
            <div className="tech-item">
              <span className="tech-icon"><Code size={20} /></span>
              <span>Python Flask</span>
            </div>
            <div className="tech-item">
              <span className="tech-icon"><Map size={20} /></span>
              <span>Leaflet Maps</span>
            </div>
            <div className="tech-item">
              <span className="tech-icon"><Brain size={20} /></span>
              <span>NumPy + Scikit-learn</span>
            </div>
            <div className="tech-item">
              <span className="tech-icon"><Image size={20} /></span>
              <span>Pillow (Image AI)</span>
            </div>
            <div className="tech-item">
              <span className="tech-icon"><Radio size={20} /></span>
              <span>WebSocket</span>
            </div>
          </div>
        </div>

        {/* Features */}
        <div className="about-card">
          <h3><Sparkles size={20} style={{marginRight: '8px', display: 'inline'}} /> Key Features</h3>
          <div className="about-list">
            <div className="about-list-item">
              <strong><Map size={16} style={{display:'inline', marginRight: '4px'}} /> Live Fire Map</strong>
              <p>Interactive world map with color-coded fire markers showing severity, confidence, and FRP.</p>
            </div>
            <div className="about-list-item">
              <strong><Camera size={16} style={{display:'inline', marginRight: '4px'}} /> Image Analysis</strong>
              <p>Upload images for AI-powered fire and smoke detection using color histograms and texture analysis.</p>
            </div>
            <div className="about-list-item">
              <strong><Bell size={16} style={{display:'inline', marginRight: '4px'}} /> Alert System</strong>
              <p>Real-time dashboard, email, and push notifications when critical fires are detected.</p>
            </div>
            <div className="about-list-item">
              <strong><BarChart2 size={16} style={{display:'inline', marginRight: '4px'}} /> Data Visualization</strong>
              <p>Interactive charts showing severity distribution, regional density, FRP analysis, and trends.</p>
            </div>
            <div className="about-list-item">
              <strong><Brain size={16} style={{display:'inline', marginRight: '4px'}} /> AI Risk Analysis</strong>
              <p>Machine learning-powered regional risk assessments with natural-language summaries.</p>
            </div>
            <div className="about-list-item">
              <strong><ClipboardList size={16} style={{display:'inline', marginRight: '4px'}} /> Fire History</strong>
              <p>Complete log of all fire detections and alerts with filtering and search capabilities.</p>
            </div>
          </div>
        </div>

        {/* Metrics Glossary */}
        <div className="about-card">
          <h3><BookOpen size={20} style={{marginRight: '8px', display: 'inline'}} /> Metrics Glossary</h3>
          <div className="about-list">
            <div className="about-list-item">
              <strong>FRP (Fire Radiative Power)</strong>
              <p>Measured in megawatts (MW). Indicates the rate of radiant heat output from a fire. Higher = more intense.</p>
            </div>
            <div className="about-list-item">
              <strong>Confidence</strong>
              <p>0–100% measure of how likely the detection is an actual fire vs false positive.</p>
            </div>
            <div className="about-list-item">
              <strong>Brightness Temperature</strong>
              <p>Measured in Kelvin (K). The thermal infrared brightness at the fire pixel.</p>
            </div>
            <div className="about-list-item">
              <strong>Severity Levels</strong>
              <p>Critical (<AlertCircle size={12} color='red' style={{display:'inline'}}/>), High (<AlertCircle size={12} color='orange' style={{display:'inline'}}/>), Moderate (<AlertCircle size={12} color='yellow' style={{display:'inline'}}/>), Low (<AlertCircle size={12} color='green' style={{display:'inline'}}/>) — based on combined FRP, confidence, and brightness scores.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
