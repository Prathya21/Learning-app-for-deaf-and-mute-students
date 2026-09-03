import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useTranslation } from '../hooks/useTranslation';

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

const CONFIDENCE_THRESHOLD = 0.3;
const SMOOTHING_WINDOW = 5;
const INFERENCE_INTERVAL_FRAMES = 8;

const GLOSS_TO_LANGUAGE = {
  HELLO: { english: 'Hello', gujarati: 'નમસ્તે' },
  THANK_YOU: { english: 'Thank you', gujarati: 'આભાર' },
  PLEASE: { english: 'Please', gujarati: 'કૃપા કરીને' },
  YES: { english: 'Yes', gujarati: 'હા' },
  NO: { english: 'No', gujarati: 'ના' },
  STUDENT: { english: 'Student', gujarati: 'વિદ્યાર્થી' },
  TEACHER: { english: 'Teacher', gujarati: 'શિક્ષક' },
  LEARN: { english: 'Learn', gujarati: 'શીખવું' },
  BOOK: { english: 'Book', gujarati: 'પુસ્તક' },
  WATER: { english: 'Water', gujarati: 'પાણી' },
  GOOD: { english: 'Good', gujarati: 'સારું' },
  BAD: { english: 'Bad', gujarati: 'ખરાબ' },
  HELP: { english: 'Help', gujarati: 'મદદ' },
  UNDERSTAND: { english: 'Understand', gujarati: 'સમજવું' },
  QUESTION: { english: 'Question', gujarati: 'પ્રશ્ન' },
  READ: { english: 'Read', gujarati: 'વાંચવા' },
  WRITE: { english: 'Write', gujarati: 'લખવા' },
  WHAT: { english: 'What', gujarati: 'શું' },
  WHERE: { english: 'Where', gujarati: 'ક્યાં' },
  HOW: { english: 'How', gujarati: 'કેવી રીતે' },
  GOODBYE: { english: 'Goodbye', gujarati: 'અવજો' },
  SORRY: { english: 'Sorry', gujarati: 'મફત કરજો' },
  OKAY: { english: 'Okay', gujarati: 'ઠીક છે' },
  ME: { english: 'Me', gujarati: 'મને' },
  YOU: { english: 'You', gujarati: 'તમે' },
  HE: { english: 'He', gujarati: 'તે (પુરુષ)' },
  SHE: { english: 'She', gujarati: 'તે (સ્ત્રી)' },
  MOTHER: { english: 'Mother', gujarati: 'માતા' },
  FATHER: { english: 'Father', gujarati: 'પિતા' },
  BROTHER: { english: 'Brother', gujarati: 'ભાઈ' },
  SISTER: { english: 'Sister', gujarati: 'બહેન' },
  FRIEND: { english: 'Friend', gujarati: 'મિત્ર' },
  SCHOOL: { english: 'School', gujarati: 'શાળા' },
  HOME: { english: 'Home', gujarati: 'ઘર' },
  HOSPITAL: { english: 'Hospital', gujarati: 'હોспи્ટલ' },
  MARKET: { english: 'Market', gujarati: 'બજાર' },
  EAT: { english: 'Eat', gujarati: 'ખાવું' },
  DRINK: { english: 'Drink', gujarati: 'પીવું' },
  FOOD: { english: 'Food', gujarati: 'ખાણું' },
  TEA: { english: 'Tea', gujarati: 'ચા' },
  COME: { english: 'Come', gujarati: 'આવવું' },
  GO: { english: 'Go', gujarati: 'જાવું' },
  SIT: { english: 'Sit', gujarati: 'બેસવા' },
  STAND: { english: 'Stand', gujarati: 'ઉભું રહેવું' },
  WHEN: { english: 'When', gujarati: 'ક્યારે' },
  TODAY: { english: 'Today', gujarati: 'આજ' }
};

const API_URL = '/api/inference/gesture';

function HandTracker() {
  const { language, setLanguage } = useTranslation();
  const [status, setStatus] = useState('Camera Off');
  const [hands, setHands] = useState([]);
  const [error, setError] = useState(null);
  const [showDebug, setShowDebug] = useState(false);
  const [inferenceResult, setInferenceResult] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [inferenceStatus, setInferenceStatus] = useState('idle');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const streamRef = useRef(null);
  const frameBufferRef = useRef([]);
  const frameCountRef = useRef(0);
  const lastInferenceTimeRef = useRef(0);
  const smoothingBufferRef = useRef([]);
  const lastGlossRef = useRef(null);

  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

  const toCanonicalFrame = useCallback((results) => {
    const frame = new Float32Array(126);
    
    // LEFT hand (indices 0-62)
    const leftHand = results.landmarks.find(h => h.handedness === 'Left');
    if (leftHand) {
      leftHand.landmarks.forEach((lm, i) => {
        frame[i * 3] = lm.x;
        frame[i * 3 + 1] = lm.y;
        frame[i * 3 + 2] = lm.z;
      });
    }
    // RIGHT hand (indices 63-125)
    const rightHand = results.landmarks.find(h => h.handedness === 'Right');
    if (rightHand) {
      rightHand.landmarks.forEach((lm, i) => {
        frame[63 + i * 3] = lm.x;
        frame[63 + i * 3 + 1] = lm.y;
        frame[63 + i * 3 + 2] = lm.z;
      });
    }
    return frame;
  }, []);

  const drawLandmarks = useCallback((ctx, landmarks, videoWidth, videoHeight) => {
    ctx.save();
    
    landmarks.forEach((hand, handIndex) => {
      const color = handIndex === 0 ? '#00FF00' : '#FF6B00';
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      HAND_CONNECTIONS.forEach(([from, to]) => {
        const p1 = hand[from];
        const p2 = hand[to];
        if (p1 && p2) {
          ctx.moveTo(p1.x * videoWidth, p1.y * videoHeight);
          ctx.lineTo(p2.x * videoWidth, p2.y * videoHeight);
        }
      });
      ctx.stroke();
      
      ctx.fillStyle = color;
      hand.forEach(point => {
        if (point) {
          ctx.beginPath();
          ctx.arc(point.x * videoWidth, point.y * videoHeight, 4, 0, 2 * Math.PI);
          ctx.fill();
        }
      });
    });
    
    ctx.restore();
  }, []);

  const runInference = useCallback(async (frames) => {
    if (inferenceStatus === 'running') return;
    setInferenceStatus('running');
    
    try {
      const response = await fetch('/api/inference/gesture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Inference failed');
      }
      
      const result = await response.json();
      return result;
    } catch (err) {
      console.error('Inference error:', err);
      return null;
    } finally {
      setInferenceStatus('idle');
    }
  }, []);

  const processFrame = useCallback(async () => {
    if (!videoRef.current || !handLandmarkerRef.current || videoRef.current.readyState !== 4) {
      animationRef.current = requestAnimationFrame(processFrame);
      return;
    }
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    try {
      const results = handLandmarkerRef.current.detectForVideo(video, performance.now());
      
      if (results.landmarks && results.landmarks.length > 0) {
        const detectedHands = results.landmarks.map((landmarks, i) => ({
          hand: results.handedness?.[i]?.[0]?.categoryName || (i === 0 ? 'Right' : 'Left'),
          confidence: results.handedness?.[i]?.[0]?.score || 1.0,
          landmarks: landmarks.map(p => ({ x: p.x, y: p.y, z: p.z }))
        }));
        
        setHands(detectedHands);
        setStatus(`${detectedHands.length} Hand${detectedHands.length !== 1 ? 's' : ''} Detected`);
        
        drawLandmarks(ctx, results.landmarks, canvas.width, canvas.height);
        
        // Convert to canonical frame
        const canonicalFrame = toCanonicalFrame(results);
        
        // Add to ring buffer
        frameBufferRef.current.push(canonicalFrame);
        if (frameBufferRef.current.length > 64) {
          frameBufferRef.current.shift();
        }
        
        frameCountRef.current++;
        
        // Run inference every INFERENCE_INTERVAL_FRAMES frames when buffer is full
        if (frameBufferRef.current.length === 64 && frameCountRef.current % 8 === 0) {
          const frames = frameBufferRef.current.slice().map(f => Array.from(f));
          const result = await runInference(frames);
          
          if (result && result.confidence >= CONFIDENCE_THRESHOLD) {
            // Temporal smoothing: add to buffer
            smoothingBufferRef.current.push(result.gloss);
            if (smoothingBufferRef.current.length > SMOOTHING_WINDOW) {
              smoothingBufferRef.current.shift();
            }
            
            // Majority vote for smoothing
            const glossCounts = {};
            smoothingBufferRef.current.forEach(g => {
              glossCounts[g] = (glossCounts[g] || 0) + 1;
            });
            const smoothedGloss = Object.entries(glossCounts).sort((a, b) => b[1] - a[1])[0]?.[0];
            
            if (smoothedGloss !== lastGlossRef.current) {
              lastGlossRef.current = smoothedGloss;
              setInferenceResult({
                gloss: smoothedGloss,
                confidence: result.confidence,
                topK: result.topK,
                modelMetadata: result.modelMetadata
              });
            }
          }
          
          const now = Date.now();
          if (showDebug && now - lastDebugUpdate.current > 500) {
            lastDebugUpdate.current = now;
          }
        } else {
          setHands([]);
          setStatus('No Hands Detected');
        }
      }
      } catch (err) {
        console.error('Detection error:', err);
      }
    
      animationRef.current = requestAnimationFrame(processFrame);
    }, [toCanonicalFrame, drawLandmarks, showDebug, inferenceStatus]);

  const initHandLandmarker = useCallback(async () => {
    setStatus('Initializing Hand Tracker...');
    setError(null);
    
    try {
      const { HandLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
      
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      
      const handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      });
      
      handLandmarkerRef.current = handLandmarker;
      setStatus('No Hands Detected');
      processFrame();
    } catch (err) {
      console.error('HandLandmarker init error:', err);
      setError('Failed to initialize hand tracking. Check console for details.');
      setStatus('Model Loading Error');
    }
  }, [processFrame]);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false
      });
      
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        await initHandLandmarker();
      }
    } catch (err) {
      console.error('Camera error:', err);
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError('Camera permission denied. Please allow camera access in browser settings.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setError('No webcam found. Please connect a camera and try again.');
      } else {
        setError(`Camera error: ${err.message}`);
      }
      setStatus('Camera Off');
    }
  }, [initHandLandmarker]);

  const stopCamera = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setHands([]);
    setInferenceResult(null);
    setPredictionHistory([]);
    setStatus('Camera Off');
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
        handLandmarkerRef.current = null;
      }
    };
  }, [stopCamera]);

  // Mirror fix: MediaPipe landmarks are already in camera coordinate space
  // The video is mirrored via CSS, but MediaPipe landmarks are already in the correct coordinate space
  // No additional mirror transformation needed for landmarks

  if (error) {
    return (
      <div className="hand-tracker error">
        <div className="status-display error">{error}</div>
        <button className="btn btn-primary" onClick={() => { setError(null); startCamera(); }}>
          Retry
        </button>
      </div>
    );
  }

  const glossTranslation = useMemo(() => {
    if (!inferenceResult?.gloss) return null;
    const translation = GLOSS_TO_LANGUAGE[inferenceResult.gloss];
    return translation ? translation[language] : inferenceResult.gloss;
  }, [inferenceResult, language]);

  return (
    <div className="hand-tracker">
      <div className="tracker-header">
        <h2>Live Hand Tracking</h2>
        <div className={`status-display ${status === 'Camera Off' ? 'off' : status.includes('Error') ? 'error' : inferenceStatus === 'running' ? 'running' : ''}`}>
          {status}
          {inferenceStatus === 'running' && <span className="inference-indicator">🔄 Inferring...</span>}
        </div>
      </div>

      <div className="camera-controls">
        {status === 'Camera Off' ? (
          <button className="btn btn-primary" onClick={startCamera} disabled={status === 'Initializing Hand Tracker...'}>
            Start Camera
          </button>
        ) : (
          <button className="btn btn-danger" onClick={stopCamera}>
            Stop Camera
          </button>
        )}
        <label className="debug-toggle">
          <input
            type="checkbox"
            checked={showDebug}
            onChange={(e) => setShowDebug(e.target.checked)}
          />
          Show Landmark Data
        </label>
        <label className="language-toggle">
          <select value={language} onChange={(e) => setLanguage(e.target.value)}>
            <option value="english">English</option>
            <option value="gujarati">ગુજરાતી</option>
          </select>
        </label>
      </div>

      <div className="status-panel">
        <div className="status-row">
          <span className={`status-indicator ${status === 'Camera Off' ? 'off' : 'on'}`}></span>
          <span>Camera: {status === 'Camera Off' ? 'OFF' : 'ACTIVE'}</span>
        </div>
        <div className="status-row">
          <span className="status-indicator">{hands.length > 0 ? '✓' : '✗'}</span>
          <span>Hands Detected: {hands.length}</span>
        </div>
        <div className="status-row">
          <span className="status-indicator">{frameBufferRef.current.length > 0 ? '✓' : '✗'}</span>
          <span>Frames Buffered: {frameBufferRef.current.length} / 64</span>
        </div>
        <div className="status-row">
          <span className="status-indicator">{inferenceStatus === 'running' ? '⟳' : '✓'}</span>
          <span>Inference: {inferenceStatus === 'running' ? 'PROCESSING...' : inferenceStatus === 'idle' ? 'READY' : inferenceStatus}</span>
        </div>
        <div className="status-row">
          <span className="status-indicator">{inferenceResult ? '✓' : '✗'}</span>
          <span>Backend: {inferenceResult ? 'CONNECTED' : 'PENDING'}</span>
        </div>
      </div>

      <div className="inference-display">
        {inferenceResult ? (
          <div className="inference-result">
            <div className="primary-gloss">
              <span className="gloss-label">{language === 'english' ? 'RECOGNIZED SIGN:' : 'અભ્યાસ:'}</span>
              <span className="gloss-text-large">{glossTranslation || inferenceResult.gloss}</span>
              <span className="confidence-badge-large">{(inferenceResult.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="top-k">
              <h4>{language === 'english' ? 'TOP PREDICTIONS:' : 'અગ્રણી ભવિષ્યવાણીઓ:'}</h4>
              {inferenceResult.topK.slice(0, 3).map((item, idx) => (
                <div key={idx} className="top-k-item">
                  <span className="rank">#{idx + 1}</span>
                  <span className="gloss">
                    {GLOSS_TO_LANGUAGE[item.gloss]?.[language] || item.gloss}
                  </span>
                  <span className="probability">{(item.probability * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="inference-placeholder">
            {inferenceStatus === 'running' ? (
              <span>🔄 Running inference...</span>
            ) : hands.length > 0 ? (
              <span>Show hands clearly for recognition</span>
            ) : (
              <span>No hands detected</span>
            )}
          </div>
        )}
      </div>

      <div className="video-wrapper">
        <video ref={videoRef} className="camera-feed" playsInline muted />
        <canvas ref={canvasRef} className="landmark-overlay" />
      </div>

      {showDebug && hands.length > 0 && (
        <div className="debug-panel">
          <h3>Landmark Data (Throttled)</h3>
          {hands.map((hand, i) => (
            <div key={i} className="hand-debug">
              <h4>{hand.hand} Hand (Confidence: {hand.confidence.toFixed(2)})</h4>
              <div className="landmark-grid">
                {hand.landmarks.slice(0, 10).map((lm, idx) => (
                  <div key={idx} className="landmark-item">
                    {idx}: ({lm.x.toFixed(3)}, {lm.y.toFixed(3)}, {lm.z.toFixed(3)})
                  </div>
                ))}
                {hand.landmarks.length > 10 && (
                  <div className="landmark-item">... and {hand.landmarks.length - 10} more</div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      <p className="tracker-note">
        <strong>Note:</strong> Hand landmark detection is not ISL gesture recognition.
        This module detects hand skeleton coordinates only.
        <br />
        <span className="dev-badge">Experimental demo model — trained on synthetic landmark data. Recognition results are not yet validated on real ISL.</span>
        <br />
        <span className="dev-badge">Confidence Threshold: {CONFIDENCE_THRESHOLD * 100}%</span>
      </p>
    </div>
  );
}

export default HandTracker;