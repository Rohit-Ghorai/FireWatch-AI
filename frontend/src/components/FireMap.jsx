import React, { useEffect, useMemo, useRef, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap, useMapEvents } from 'react-leaflet';
import { AlertCircle, AlertTriangle } from 'lucide-react';
import { GoogleMap, MarkerF, useJsApiLoader } from '@react-google-maps/api';
import 'leaflet/dist/leaflet.css';

// OpenStreetMap tile configuration — no API key required.
// Using the official OSM tile server per https://wiki.openstreetmap.org/wiki/Tile_usage_policy
// Attribution is mandatory: © OpenStreetMap contributors
const OSM_TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

// Stadia Maps dark theme — free tier, no key needed for development/low-traffic
// Docs: https://docs.stadiamaps.com/
const STADIA_DARK_TILE_URL = 'https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}{r}.png';
const STADIA_ATTRIBUTION = '&copy; <a href="https://stadiamaps.com/">Stadia Maps</a>, &copy; <a href="https://openmaptiles.org/">OpenMapTiles</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

const severityColors = {
  critical: '#ef4444',
  high: '#f97316',
  moderate: '#f59e0b',
  low: '#22c55e',
};

const severityRadius = {
  critical: 12,
  high: 10,
  moderate: 8,
  low: 6,
};

// --- Tile error handler: logs tile failures separately from NASA data status ---
function TileErrorLogger() {
  useMapEvents({
    tileerror: (e) => {
      console.warn('[Map] Tile load failed:', e.tile?.src || 'unknown tile URL');
    },
  });
  return null;
}

function MapUpdater({ fires }) {
  useMap();
  const hasZoomed = useRef(false);

  useEffect(() => {
    if (fires.length > 0 && !hasZoomed.current) {
      // Don't auto-zoom — keep world view
      hasZoomed.current = true;
    }
  }, [fires]);

  return null;
}

// Shared fire marker layer used by both Google Maps and Leaflet fallback
function FireMarkers({ fires, heatPoints, viewMode }) {
  return (viewMode === 'markers' ? fires : heatPoints).map((fire, idx) => (
    <CircleMarker
      key={fire.id || idx}
      center={[fire.latitude, fire.longitude]}
      radius={viewMode === 'markers' ? (severityRadius[fire.severity] || 5) : Math.max(8, Math.round((fire.heatWeight || 0.2) * 20))}
      pathOptions={{
        fillColor: viewMode === 'markers' ? (severityColors[fire.severity] || '#f59e0b') : '#ef4444',
        fillOpacity: viewMode === 'markers' ? 0.8 : Math.max(0.12, (fire.heatWeight || 0.2) * 0.7),
        color: viewMode === 'markers' ? 'rgba(255,255,255,0.6)' : 'rgba(249,115,22,0.0)',
        weight: viewMode === 'markers' ? 2 : 0,
      }}
    >
      {viewMode === 'markers' && (
        <Popup>
          <div>
            <div className="popup-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <AlertCircle size={16} color={severityColors[fire.severity]} />
              {fire.is_simulated ? 'Fire Detection (Simulated)' : 'Fire Detection (NASA FIRMS)'}
            </div>
            <div className="popup-row">
              <span className="popup-label">Data Source</span>
              <span className="popup-value" style={{ fontWeight: 600, color: fire.is_simulated ? '#f59e0b' : '#38bdf8' }}>
                {fire.is_simulated ? 'SIMULATED (Demo)' : 'NASA FIRMS'}
              </span>
            </div>
            <div className="popup-row">
              <span className="popup-label">Location</span>
              <span className="popup-value">{fire.latitude?.toFixed(3)}, {fire.longitude?.toFixed(3)}</span>
            </div>
            <div className="popup-row">
              <span className="popup-label">Confidence</span>
              <span className="popup-value">{fire.confidence}%</span>
            </div>
            <div className="popup-row">
              <span className="popup-label">FRP</span>
              <span className="popup-value">{fire.frp} MW</span>
            </div>
            <div className="popup-row">
              <span className="popup-label">Brightness</span>
              <span className="popup-value">{fire.brightness} K</span>
            </div>
            <div className="popup-row">
              <span className="popup-label">Satellite</span>
              <span className="popup-value">{fire.satellite}</span>
            </div>
            <div className="popup-row">
              <span className="popup-label">Time</span>
              <span className="popup-value">{fire.acq_date} {fire.acq_time}</span>
            </div>
            <div className="popup-row">
              <span className="popup-label">Severity</span>
              <span className="popup-value" style={{ color: severityColors[fire.severity] }}>
                {fire.severity?.toUpperCase()}
              </span>
            </div>
          </div>
        </Popup>
      )}
    </CircleMarker>
  ));
}

export default function FireMap({ fires }) {
  const [viewMode, setViewMode] = useState('markers');
  const googleApiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
  const { isLoaded } = useJsApiLoader({
    id: 'google-map-script',
    googleMapsApiKey: googleApiKey || '',
  });

  const heatPoints = useMemo(() => {
    return fires.map((f) => ({
      ...f,
      heatWeight: Math.min(1, (f.frp || 0) / 120 + 0.08),
    }));
  }, [fires]);

  // --- Google Maps path (only when a valid API key is configured) ---
  if (googleApiKey && isLoaded) {
    return (
      <div className="map-container">
        <div className="map-mode-toggle">
          <button className={`htab ${viewMode === 'markers' ? 'active' : ''}`} onClick={() => setViewMode('markers')}>Markers</button>
          <button className={`htab ${viewMode === 'heatmap' ? 'active' : ''}`} onClick={() => setViewMode('heatmap')}>Heatmap</button>
        </div>
        <GoogleMap
          mapContainerStyle={{ width: '100%', height: '100%' }}
          center={{ lat: 20, lng: 0 }}
          zoom={2}
          options={{
            disableDefaultUI: true,
            zoomControl: true,
            styles: [
              { elementType: 'geometry', stylers: [{ color: '#0f172a' }] },
              { elementType: 'labels.text.stroke', stylers: [{ color: '#0f172a' }] },
              { elementType: 'labels.text.fill', stylers: [{ color: '#64748b' }] },
            ],
          }}
        >
          {viewMode === 'markers' && fires.map((fire, idx) => (
            <MarkerF
              key={fire.id || idx}
              position={{ lat: fire.latitude, lng: fire.longitude }}
              title={`${fire.is_simulated ? '[SIMULATED] ' : '[NASA] '}${fire.severity?.toUpperCase()} • FRP ${fire.frp} MW`}
              icon={{
                path: window.google.maps.SymbolPath.CIRCLE,
                fillColor: severityColors[fire.severity] || '#f59e0b',
                fillOpacity: 0.9,
                strokeColor: '#ffffff',
                strokeWeight: 1,
                scale: severityRadius[fire.severity] || 5,
              }}
            />
          ))}

          {viewMode === 'heatmap' && heatPoints.map((fire, idx) => (
            <MarkerF
              key={`heat_${fire.id || idx}`}
              position={{ lat: fire.latitude, lng: fire.longitude }}
              icon={{
                path: window.google.maps.SymbolPath.CIRCLE,
                fillColor: '#ef4444',
                fillOpacity: Math.max(0.1, fire.heatWeight * 0.7),
                strokeColor: '#f97316',
                strokeWeight: 0,
                scale: Math.max(6, Math.round(fire.heatWeight * 18)),
              }}
            />
          ))}
        </GoogleMap>
      </div>
    );
  }

  // --- Leaflet fallback: OpenStreetMap tiles (no API key required) ---
  // Primary: Stadia Maps dark theme (matches dashboard dark style)
  // Fallback: standard OpenStreetMap tiles
  return (
    <div className="map-container">
      <div className="map-mode-toggle">
        <button className={`htab ${viewMode === 'markers' ? 'active' : ''}`} onClick={() => setViewMode('markers')}>Markers</button>
        <button className={`htab ${viewMode === 'heatmap' ? 'active' : ''}`} onClick={() => setViewMode('heatmap')}>Heatmap</button>
      </div>

      <MapContainer
        center={[20, 0]}
        zoom={2}
        minZoom={2}
        maxZoom={18}
        style={{ width: '100%', height: '100%' }}
        zoomControl={true}
        attributionControl={true}
      >
        {/* Primary dark tile layer — Stadia Maps, no key required for ≤5k req/day */}
        <TileLayer
          url={STADIA_DARK_TILE_URL}
          attribution={STADIA_ATTRIBUTION}
          maxZoom={20}
          subdomains="abcd"
          detectRetina={true}
          errorTileUrl=""
          crossOrigin="anonymous"
          eventHandlers={{
            tileerror: (e) => {
              // Log tile errors independently — do NOT confuse with NASA data status
              console.warn('[Map] Stadia tile failed, check network connectivity. NASA data status is unaffected.', e);
            },
          }}
        />

        {/* OSM fallback tile layer — renders under Stadia in case Stadia tiles fail to load */}
        <TileLayer
          url={OSM_TILE_URL}
          attribution={OSM_ATTRIBUTION}
          opacity={0.08}
          maxZoom={19}
          detectRetina={true}
          errorTileUrl=""
          crossOrigin="anonymous"
        />

        <TileErrorLogger />
        <MapUpdater fires={fires} />

        {/* Zero-detection empty state — clearly separate from map tile status */}
        {fires.length === 0 && (
          <div
            style={{
              position: 'absolute',
              bottom: '20px',
              left: '50%',
              transform: 'translateX(-50%)',
              zIndex: 1000,
              background: 'rgba(15,23,42,0.88)',
              border: '1px solid rgba(99,102,241,0.3)',
              borderRadius: '8px',
              padding: '10px 18px',
              color: '#94a3b8',
              fontSize: '13px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              pointerEvents: 'none',
              backdropFilter: 'blur(8px)',
              whiteSpace: 'nowrap',
            }}
          >
            <AlertTriangle size={14} style={{ color: '#f59e0b', flexShrink: 0 }} />
            No active NASA fire detections in the current observation window
          </div>
        )}

        <FireMarkers fires={fires} heatPoints={heatPoints} viewMode={viewMode} />
      </MapContainer>
    </div>
  );
}
