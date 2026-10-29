import React, { useState, useEffect, useRef, useCallback } from 'react';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import CameraPanel from './components/CameraPanel';
import CaptionPanel from './components/CaptionPanel';
import TranslationControls from './components/TranslationControls';
import DemoPresenterModal from './components/DemoPresenterModal';
import SignLibraryModal from './components/SignLibraryModal';
import HowItWorks from './components/HowItWorks';
import UseCases from './components/UseCases';
import Roadmap from './components/Roadmap';
import HistoryDrawer from './components/HistoryDrawer';
import SettingsModal from './components/SettingsModal';
import Footer from './components/Footer';
import InfoPage from './components/InfoPage';

import { WebcamManager } from './utils/webcamManager';
import { HandLandmarkDetector } from './utils/handLandmarkDetector';
import { PoseLandmarkDetector } from './utils/poseLandmarkDetector';
import {
  HeuristicISLClassifier,
  SimulatedDemoClassifier,
  KaggleISLModelClassifier,
  buildHolisticFrame,
} from './utils/signClassifierAdapter';
import { TranscriptManager } from './utils/transcriptManager';
import { SpeechSynthesizer } from './utils/speechSynthesizer';
import { getAllPhrases } from './utils/demoData';

// Recognition engine preferences persist across reloads (the Kaggle URL is tedious to re-paste)
const ENGINE_STORAGE_KEY = 'learnable_recognition_engine';
const SERVER_URL_STORAGE_KEY = 'learnable_model_server_url';

function readStoredSetting(key, fallback) {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}

export default function App() {
  const pageName = new URLSearchParams(window.location.search).get('page') || 'home';
  // Navigation & Modal States
  const [activeSection, setActiveSection] = useState('hero');
  const [isDemoModalOpen, setIsDemoModalOpen] = useState(() => new URLSearchParams(window.location.search).get('demo') === '1');
  const [isLibraryOpen, setIsLibraryOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);

  // Vocabulary list (built-in + user created custom signs)
  const [allPhrases, setAllPhrases] = useState(() => getAllPhrases());

  const refreshPhrases = useCallback(() => {
    setAllPhrases(getAllPhrases());
  }, []);

  // Display preferences
  const [highContrast, setHighContrast] = useState(false);
  const [captionFontSize, setCaptionFontSize] = useState('normal');
  const [selectedLanguage, setSelectedLanguage] = useState('ISL');
  const [selectedDomain, setSelectedDomain] = useState('all');

  // Camera & Vision States
  const [cameraStatus, setCameraStatus] = useState('idle'); // 'idle' | 'initializing' | 'live' | 'error'
  const [errorMessage, setErrorMessage] = useState(null);
  const [isHandDetected, setIsHandDetected] = useState(false);
  const [handCount, setHandCount] = useState(0);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [fps, setFps] = useState(30);

  // Settings states
  const [videoDevices, setVideoDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [resolution, setResolution] = useState('720p');
  const [overlayColor, setOverlayColor] = useState('cyan');
  const [speechRate, setSpeechRate] = useState(1.0);

  // Recognition engine: on-device heuristic gestures or the pretrained ISL model on Kaggle
  const [recognitionEngine, setRecognitionEngine] = useState(() => readStoredSetting(ENGINE_STORAGE_KEY, 'heuristic'));
  const [modelServerUrl, setModelServerUrl] = useState(() => readStoredSetting(SERVER_URL_STORAGE_KEY, ''));
  const [modelStatus, setModelStatus] = useState(null);

  // Captions & Translation State
  const [currentCaption, setCurrentCaption] = useState('Hello, welcome to Learnable!');
  const [hindiCaption, setHindiCaption] = useState('नमस्ते, लर्नेबल में आपका स्वागत है!');
  const [islGloss, setIslGloss] = useState('NAMASTE / WELCOME');
  const [confidence, setConfidence] = useState(0.96);
  const [timestamp, setTimestamp] = useState(TranscriptManager.formatTimestamp());
  const [source, setSource] = useState('demo');
  const [sourceLabel, setSourceLabel] = useState('Interactive Prototype Demo');

  // Session Transcript History
  const [history, setHistory] = useState([
    {
      id: 'init-1',
      text: 'Hello, welcome to Learnable!',
      hindiText: 'नमस्ते, लर्नेबल में आपका स्वागत है!',
      islGloss: 'NAMASTE / WELCOME',
      confidence: 0.96,
      timestamp: TranscriptManager.formatTimestamp(),
      source: 'demo',
      domain: 'greetings',
    }
  ]);

  // DOM Refs
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const webcamManagerRef = useRef(null);
  const detectorRef = useRef(null);
  const poseDetectorRef = useRef(null);
  const islModelRef = useRef(null);
  const recognitionEngineRef = useRef(recognitionEngine);
  const animFrameIdRef = useRef(null);
  const lastAddCaptionTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsCheckTimeRef = useRef(performance.now());

  /**
   * Show a live recognition result (heuristic gesture or pretrained model) and log it
   */
  const showLiveMatch = (match, domain) => {
    const ts = TranscriptManager.formatTimestamp();

    setCurrentCaption(match.text);
    setHindiCaption(match.hindiText || '');
    setIslGloss(match.islGloss || '');
    setConfidence(match.confidence);
    setTimestamp(ts);
    setSource(match.source);
    setSourceLabel(match.sourceLabel);

    // Add to session transcript history
    setHistory(prev => [
      ...prev,
      {
        id: 'hist-' + Date.now(),
        text: match.text,
        hindiText: match.hindiText,
        islGloss: match.islGloss,
        confidence: match.confidence,
        timestamp: ts,
        source: match.source,
        domain: domain || 'general',
      }
    ]);
  };

  // Instantiate Singletons
  useEffect(() => {
    if (pageName !== 'home') return undefined;
    webcamManagerRef.current = new WebcamManager();
    detectorRef.current = new HandLandmarkDetector();
    poseDetectorRef.current = new PoseLandmarkDetector();
    islModelRef.current = new KaggleISLModelClassifier({
      onPrediction: (match) => showLiveMatch(match, match.domain),
      onStatusChange: setModelStatus,
    });

    // Check available cameras
    WebcamManager.getVideoDevices().then((devices) => {
      setVideoDevices(devices);
    });

    return () => {
      stopCameraFlow();
      if (detectorRef.current) {
        detectorRef.current.destroy();
      }
      poseDetectorRef.current.destroy();
      islModelRef.current.reset();
      SpeechSynthesizer.stop();
    };
  }, []);

  // Keep the Kaggle model client in sync with the selected engine and server URL
  useEffect(() => {
    recognitionEngineRef.current = recognitionEngine;
    islModelRef.current.reset();

    // The body pose tracker is only needed (and downloaded) for the pretrained model
    if (recognitionEngine === 'kaggle' && cameraStatus === 'live') {
      poseDetectorRef.current.initialize().catch((err) => {
        console.error('PoseLandmarker failed to load:', err);
        setModelStatus({ state: 'error', message: 'Could not load the body pose tracker' });
      });
    }
  }, [recognitionEngine, cameraStatus]);

  useEffect(() => {
    islModelRef.current.setServerUrl(modelServerUrl);
  }, [modelServerUrl]);

  useEffect(() => {
    try {
      localStorage.setItem(ENGINE_STORAGE_KEY, recognitionEngine);
      localStorage.setItem(SERVER_URL_STORAGE_KEY, modelServerUrl);
    } catch {
      // Storage unavailable (e.g. private mode); settings just won't persist
    }
  }, [recognitionEngine, modelServerUrl]);

  // Sync High Contrast class to html body
  useEffect(() => {
    if (highContrast) {
      document.documentElement.classList.add('high-contrast');
    } else {
      document.documentElement.classList.remove('high-contrast');
    }
  }, [highContrast]);

  // Handle resolution dimensions
  const getResolutionDimensions = () => {
    switch (resolution) {
      case '1080p': return { width: 1920, height: 1080 };
      case '480p': return { width: 640, height: 480 };
      case '720p':
      default: return { width: 1280, height: 720 };
    }
  };

  // Overlay color palettes
  const getOverlayColors = () => {
    switch (overlayColor) {
      case 'violet':
        return { primaryColor: '#A78BFA', secondaryColor: '#C4B5FD', accentColor: '#FFFFFF' };
      case 'emerald':
        return { primaryColor: '#34D399', secondaryColor: '#6EE7B7', accentColor: '#FFFFFF' };
      case 'cyan':
      default:
        return { primaryColor: '#22D3EE', secondaryColor: '#A78BFA', accentColor: '#FFFFFF' };
    }
  };

  /**
   * Main Real-Time Detection Loop
   */
  const runDetectionLoop = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const detector = detectorRef.current;

    if (!video || !canvas || !detector || !detector.isReady) {
      animFrameIdRef.current = requestAnimationFrame(runDetectionLoop);
      return;
    }

    // Measure FPS
    frameCountRef.current++;
    const now = performance.now();
    if (now - lastFpsCheckTimeRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsCheckTimeRef.current = now;
    }

    // Match canvas dimensions to video display dimensions
    if (video.videoWidth && canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // Extract landmarks from current video frame
    const result = detector.detectForVideo(video);

    // Pretrained ISL model: stream body + hand landmarks of each new frame (never video)
    const useKaggleModel = recognitionEngineRef.current === 'kaggle';
    if (result && useKaggleModel && poseDetectorRef.current.isReady) {
      const pose = poseDetectorRef.current.detectForVideo(video);
      islModelRef.current.pushFrame(
        buildHolisticFrame(pose, result.landmarks, result.handednesses),
        result.landmarks.length > 0,
        now
      );
    }

    if (result && result.landmarks && result.landmarks.length > 0) {
      setIsHandDetected(true);
      setHandCount(result.landmarks.length);

      // Draw skeleton on canvas
      if (showLandmarks) {
        HandLandmarkDetector.drawSkeleton(
          canvas,
          result.landmarks,
          getOverlayColors()
        );
      } else {
        const ctx = canvas.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
      }

      // Check heuristic sign match on primary hand (the pretrained model reports asynchronously)
      const primaryLandmarks = result.landmarks[0];
      const match = useKaggleModel ? null : HeuristicISLClassifier.evaluateHand(primaryLandmarks);

      if (match && match.hasMatch) {
        const timeNow = Date.now();
        // Prevent flood: debounce 2.2 seconds between automatic live detections
        if (timeNow - lastAddCaptionTimeRef.current > 2200) {
          lastAddCaptionTimeRef.current = timeNow;
          showLiveMatch(match, selectedDomain);
        }
      }

    } else {
      setIsHandDetected(false);
      setHandCount(0);
      const ctx = canvas.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    animFrameIdRef.current = requestAnimationFrame(runDetectionLoop);
  }, [showLandmarks, overlayColor, selectedDomain]);

  /**
   * Start Camera flow with explicit user consent
   */
  const startCameraFlow = async () => {
    setCameraStatus('initializing');
    setErrorMessage(null);

    try {
      const dimensions = getResolutionDimensions();
      await webcamManagerRef.current.startStream({
        deviceId: selectedDeviceId || undefined,
        width: dimensions.width,
        height: dimensions.height,
      });

      if (videoRef.current) {
        await webcamManagerRef.current.attachToVideo(videoRef.current);
      }

      // Initialize MediaPipe Vision Task
      const visionLoaded = await detectorRef.current.initialize();
      if (!visionLoaded) {
        console.warn('Vision task using heuristic/fallback mode.');
      }

      setCameraStatus('live');
      // Kick off continuous animation loop
      animFrameIdRef.current = requestAnimationFrame(runDetectionLoop);
    } catch (err) {
      console.error('Camera startup error:', err);
      setCameraStatus('error');
      setErrorMessage(err.userMessage || err.message || 'Could not access camera.');
    }
  };

  /**
   * Stop Camera and release hardware
   */
  const stopCameraFlow = () => {
    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }

    if (webcamManagerRef.current) {
      webcamManagerRef.current.stopStream();
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    setIsHandDetected(false);
    setHandCount(0);
    setCameraStatus('idle');
  };


  /**
   * Trigger sample demo phrase (from demo suite or sign library)
   */
  const handleTriggerDemoPhrase = (phrase) => {
    const formatted = SimulatedDemoClassifier.formatDemoResult(phrase);
    const ts = TranscriptManager.formatTimestamp();

    setCurrentCaption(formatted.text);
    setHindiCaption(formatted.hindiText || '');
    setIslGloss(formatted.islGloss || '');
    setConfidence(formatted.confidence);
    setTimestamp(ts);
    setSource(formatted.source);
    setSourceLabel(formatted.sourceLabel);

    // Add to session transcript history
    setHistory(prev => [
      ...prev,
      {
        id: 'demo-' + Date.now(),
        text: formatted.text,
        hindiText: formatted.hindiText,
        islGloss: formatted.islGloss,
        confidence: formatted.confidence,
        timestamp: ts,
        source: 'demo',
        domain: phrase.domain || 'general',
      }
    ]);

    // Speak caption with speech synthesizer
    SpeechSynthesizer.speak(formatted.text, {
      rate: speechRate || 1.0,
      pitch: 1.0,
    });
  };

  /**
   * Clear active captions
   */
  const handleClearCaptions = () => {
    setCurrentCaption('');
    setHindiCaption('');
    setIslGloss('');
    setConfidence(null);
    setSource('idle');
    setSourceLabel('Captions cleared');
  };

  /**
   * Reset entire session
   */
  const handleResetSession = () => {
    handleClearCaptions();
    setHistory([]);
    SpeechSynthesizer.stop();
  };

  /**
   * Download formatted transcript
   */
  const handleDownloadTranscript = () => {
    TranscriptManager.downloadTranscriptFile(history, {
      language: selectedLanguage,
      mode: cameraStatus === 'live' ? 'Live Camera Vision' : 'Interactive Prototype Demo',
    });
  };

  /**
   * Copy to clipboard helper
   */
  const handleCopyText = async (text) => {
    return await TranscriptManager.copyToClipboard(text);
  };

  /**
   * Jump to scenario from use case section
   */
  const handleSelectScenario = (domainId) => {
    setSelectedDomain(domainId);
    const sample = allPhrases.find(p => p.domain === domainId) || allPhrases[0];
    handleTriggerDemoPhrase(sample);

    const el = document.getElementById('workspace');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  if (pageName !== 'home') return <InfoPage page={pageName} />;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 flex flex-col font-sans selection:bg-violet-200 selection:text-slate-900">
      
      {/* Navigation Bar */}
      <Navbar
        onOpenDemo={() => setIsDemoModalOpen(true)}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onOpenLibrary={() => setIsLibraryOpen(true)}
        activeSection={activeSection}
        setActiveSection={setActiveSection}
        signCount={allPhrases.length}
      />

      {/* Main Content Area */}
      <main className="flex-1">
        
        {/* Hero Section */}
        <HeroSection
          onTryDemo={() => setIsDemoModalOpen(true)}
        />

        {/* Translation Studio Workspace Section */}
        <section id="workspace" className="py-8 sm:py-12 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto scroll-mt-20">
          
          {/* Section Header */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-6">
            <div>
              <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-md bg-cyan-950/60 text-cyan-300 border border-cyan-500/30 text-xs font-semibold uppercase tracking-wider mb-1">
                Real-Time Workspace
              </div>
              <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
                Live Translation Studio
              </h2>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setIsLibraryOpen(true)}
                className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-navy-850 hover:bg-navy-800 text-cyan-300 border border-cyan-500/30 flex items-center gap-1.5 shadow-sm cursor-pointer"
              >
                <span>Sign Library ({allPhrases.length})</span>
              </button>

              <button
                onClick={() => setIsDemoModalOpen(true)}
                className="px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-violet-950/80 hover:bg-violet-900 text-violet-200 border border-violet-500/30 flex items-center gap-1.5 shadow-sm cursor-pointer"
              >
                <span>Launch Demo Suite</span>
              </button>
            </div>
          </div>

          {/* Translation Controls Bar */}
          <div className="mb-6">
            <TranslationControls
              selectedLanguage={selectedLanguage}
              setSelectedLanguage={setSelectedLanguage}
              selectedDomain={selectedDomain}
              setSelectedDomain={setSelectedDomain}
              captionFontSize={captionFontSize}
              setCaptionFontSize={setCaptionFontSize}
              onResetSession={handleResetSession}
              onToggleHistory={() => setIsHistoryOpen(!isHistoryOpen)}
              onOpenLibrary={() => setIsLibraryOpen(true)}
              historyCount={history.length}
              signCount={allPhrases.length}
            />
          </div>

          {/* Two-Panel Workspace Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
            
            {/* Left Panel: Camera Studio */}
            <div className="h-full">
              <CameraPanel
                cameraStatus={cameraStatus}
                errorMessage={errorMessage}
                isHandDetected={isHandDetected}
                handCount={handCount}
                showLandmarks={showLandmarks}
                setShowLandmarks={setShowLandmarks}
                onStartCamera={startCameraFlow}
                onStopCamera={stopCameraFlow}
                onOpenSettings={() => setIsSettingsOpen(true)}
                videoRef={videoRef}
                canvasRef={canvasRef}
                fps={fps}
                recognitionEngine={recognitionEngine}
                modelStatus={modelStatus}
              />
            </div>

            {/* Right Panel: Live Captions */}
            <div className="h-full">
              <CaptionPanel
                currentCaption={currentCaption}
                hindiCaption={hindiCaption}
                islGloss={islGloss}
                confidence={confidence}
                timestamp={timestamp}
                source={source}
                sourceLabel={sourceLabel}
                captionFontSize={captionFontSize}
                onClearCaptions={handleClearCaptions}
                onCopyCaptions={handleCopyText}
                onDownloadTranscript={handleDownloadTranscript}
                onOpenHistory={() => setIsHistoryOpen(true)}
                historyCount={history.length}
              />
            </div>

          </div>

        </section>

        {/* How It Works Section */}
        <HowItWorks />

        {/* Use Cases Section */}
        <UseCases onSelectScenario={handleSelectScenario} />

        {/* Progress & Roadmap Section */}
        <Roadmap />

      </main>

      {/* Footer */}
      <Footer onOpenDemo={() => setIsDemoModalOpen(true)} />

      {/* Sign Library & Custom Sign Creation Modal */}
      <SignLibraryModal
        isOpen={isLibraryOpen}
        onClose={() => setIsLibraryOpen(false)}
        allPhrases={allPhrases}
        onRefreshPhrases={refreshPhrases}
        onSelectPhrase={handleTriggerDemoPhrase}
      />

      {/* Presenter Demo Suite Modal */}
      <DemoPresenterModal
        isOpen={isDemoModalOpen}
        onClose={() => setIsDemoModalOpen(false)}
        onTriggerDemoPhrase={handleTriggerDemoPhrase}
        onResetDemo={handleResetSession}
        onOpenLibrary={() => {
          setIsDemoModalOpen(false);
          setIsLibraryOpen(true);
        }}
        allPhrases={allPhrases}
        currentCaption={currentCaption}
      />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        videoDevices={videoDevices}
        selectedDeviceId={selectedDeviceId}
        setSelectedDeviceId={setSelectedDeviceId}
        overlayColor={overlayColor}
        setOverlayColor={setOverlayColor}
        speechRate={speechRate}
        setSpeechRate={setSpeechRate}
        resolution={resolution}
        setResolution={setResolution}
        recognitionEngine={recognitionEngine}
        setRecognitionEngine={setRecognitionEngine}
        modelServerUrl={modelServerUrl}
        setModelServerUrl={setModelServerUrl}
      />

      {/* Session History Drawer */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        history={history}
        onClearHistory={() => setHistory([])}
        onDownloadTranscript={handleDownloadTranscript}
        onCopyPhrase={handleCopyText}
      />

    </div>
  );
}
