import React, { useState, useCallback, useEffect, useRef } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { Flame, Map, Camera, History, Globe, BarChart2, BellRing, Info, Video, Activity, Menu, X } from 'lucide-react';

const NAV_ITEMS = [
  { to: '/', icon: Map,       label: 'Dashboard',        end: true },
  { to: '/upload',          icon: Camera,    label: 'Footage Analysis' },
  { to: '/live-feed',       icon: Video,     label: 'Live Camera' },
  { to: '/sensors',         icon: Activity,  label: 'Sensors' },
  { to: '/history',         icon: History,   label: 'History' },
  { to: '/regions',         icon: Globe,     label: 'Regions' },
  { to: '/visualizations',  icon: BarChart2, label: 'Charts' },
  { to: '/alert-center',    icon: BellRing,  label: 'Alert Center' },
  { to: '/about',           icon: Info,      label: 'About' },
];

export default function Header({ lastUpdated, isFetching, totalFires, socketStatus, onRefresh, voiceEnabled, onToggleVoice, onTestVoice, firmsMeta }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const navRef = useRef(null);
  const btnRef = useRef(null);
  const location = useLocation();

  const syncLabel = {
    connected: 'Live sync',
    connecting: 'Syncing',
    degraded: 'Reconnect',
    offline: 'Offline',
  }[socketStatus] || 'Syncing';

  const isDemo = firmsMeta?.isDemoData ?? true;
  const satelliteName = firmsMeta?.satelliteId || firmsMeta?.satellite || 'VIIRS_SNPP_NRT';
  const apiStatus = firmsMeta?.apiStatus || 'unknown';

  // Close mobile menu on route change
  useEffect(() => { setMenuOpen(false); }, [location.pathname]);

  // Close mobile menu when clicking outside
  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e) => {
      if (navRef.current?.contains(e.target) || btnRef.current?.contains(e.target)) return;
      setMenuOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [menuOpen]);

  const toggleMenu = useCallback(() => setMenuOpen(o => !o), []);

  return (
    <header className="header">
      <div className="header-brand">
        <div className="header-logo"><Flame size={24} /></div>
        <div>
          <div className="header-title">FireWatch AI</div>
          <div className="header-subtitle">NASA Satellite Fire Detection</div>
        </div>
      </div>

      {/* Hamburger — visible only on mobile via CSS */}
      <button
        ref={btnRef}
        className="header-menu-btn"
        onClick={toggleMenu}
        aria-label={menuOpen ? 'Close menu' : 'Open menu'}
        aria-expanded={menuOpen}
      >
        {menuOpen ? <X size={22} /> : <Menu size={22} />}
      </button>

      {/* Nav — always present on desktop; toggled on mobile */}
      <nav ref={navRef} className={`header-nav${menuOpen ? ' open' : ''}`}>
        {NAV_ITEMS.map(({ to, icon: Icon, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
          >
            <Icon size={18} /> {label}
          </NavLink>
        ))}
      </nav>

      <div className="header-status">
        <div
          className={`status-indicator source ${isDemo ? 'demo' : 'nasa'}`}
          title={isDemo ? `Data Source: Simulated / Demo (Status: ${apiStatus})` : `Data Source: NASA FIRMS (${satelliteName}) - Verified`}
        >
          <span className="status-dot"></span>
          {isDemo ? 'DEMO DATA' : 'NASA FIRMS'}
        </div>

        <div className="status-indicator">
          <span className={`status-dot ${isFetching ? 'fetching' : ''}`}></span>
          {isFetching ? 'Fetching...' : isDemo ? `${totalFires} fires (simulated)` : `${totalFires} NASA fires`}
        </div>

        <div className={`status-indicator sync ${socketStatus || 'connecting'}`}>
          <span className="status-dot"></span>
          {syncLabel}
        </div>

        <button className={`refresh-btn ${isFetching ? 'spinning' : ''}`} onClick={onRefresh}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M23 4v6h-6M1 20v-6h6" />
            <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
          </svg>
          Refresh
        </button>
      </div>
    </header>
  );
}
