import React, { useState, useEffect, useRef, useCallback } from 'react';

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];

function HandTracker() {
  const [status, setStatus] = useState('Camera Off');
  const [hands, setHands] = useState([]);
  const [error, setError] = useState(null);
  const [showDebug, setShowDebug] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const animationRef = useRef(null);
  const streamRef = useRef(null);
  const lastDebugUpdate = useRef(0);

  const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

  const drawLandmarks = useCallback((ctx, landmarks, videoWidth, videoHeight) => {
    ctx.save();
    ctx.scale(1, 1);

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
          landmarks: landmarks.map(p => ({
            x: p.x,
            y: p.y,
            z: p.z
          }))
        }));

        setHands(detectedHands);
        setStatus(`${detectedHands.length} Hand${detectedHands.length !== 1 ? 's' : ''} Detected`);

        drawLandmarks(ctx, results.landmarks, canvas.width, canvas.height);

        const now = Date.now();
        if (showDebug && now - lastDebugUpdate.current > 500) {
          lastDebugUpdate.current = now;
        }
      } else {
        setHands([]);
        setStatus('No Hands Detected');
      }
    } catch (err) {
      console.error('Detection error:', err);
    }

    animationRef.current = requestAnimationFrame(processFrame);
  }, [drawLandmarks, showDebug]);

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

  return (
    <div className="hand-tracker">
      <div className="tracker-header">
        <h2>Live Hand Tracking</h2>
        <div className={`status-display ${status === 'Camera Off' ? 'off' : status.includes('Error') ? 'error' : ''}`}>
          {status}
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
      </p>
    </div>
  );
}

export default HandTracker;