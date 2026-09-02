import React, { useState, useRef, useEffect, useCallback } from 'react';

const PLAYBACK_SPEEDS = [0.5, 0.75, 1, 1.25, 1.5];

function ISLVideoPlayer({ videos = [], glossSequence = [], onEnd }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const [videoError, setVideoError] = useState(null);
  const [showControls, setShowControls] = useState(true);
  const [hasUserInteracted, setHasUserInteracted] = useState(false);

  const videoRef = useRef(null);
  const availableVideos = videos.filter(v => v.found && v.video_url);

  const getVideoStatus = useCallback((index) => {
    const video = videos[index];
    if (!video) return 'upcoming';
    if (!video.found) return 'unavailable';
    if (index < currentIndex) return 'played';
    if (index === currentIndex) return 'playing';
    return 'upcoming';
  }, [videos, currentIndex]);

  const playVideoAtIndex = useCallback(async (index) => {
    const video = availableVideos[index];
    if (!video || !videoRef.current) return;

    setVideoError(null);
    setCurrentIndex(index);
    
    try {
      videoRef.current.src = video.video_url;
      videoRef.current.playbackRate = playbackSpeed;
      await videoRef.current.play();
      setIsPlaying(true);
      setHasUserInteracted(true);
    } catch (err) {
      console.error('Video play error:', err);
      setVideoError(`Failed to play "${video.word}": ${err.message}`);
      handleVideoError();
    }
  }, [availableVideos, playbackSpeed]);

  const handleVideoEnd = useCallback(() => {
    const nextIndex = currentIndex + 1;
    if (nextIndex < availableVideos.length) {
      playVideoAtIndex(nextIndex);
    } else {
      setIsPlaying(false);
      if (onEnd) onEnd();
    }
  }, [currentIndex, availableVideos.length, playVideoAtIndex, onEnd]);

  const handleVideoError = useCallback(() => {
    setVideoError(`Video for "${availableVideos[currentIndex]?.word}" unavailable`);
    const nextIndex = currentIndex + 1;
    if (nextIndex < availableVideos.length) {
      setTimeout(() => playVideoAtIndex(nextIndex), 500);
    } else {
      setIsPlaying(false);
    }
  }, [currentIndex, availableVideos, playVideoAtIndex]);

  const handlePlayPause = useCallback(() => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      if (!hasUserInteracted) setHasUserInteracted(true);
      videoRef.current.play().catch(() => {
        setVideoError('Autoplay blocked. Click Play to start.');
      });
    }
  }, [isPlaying, hasUserInteracted]);

  const handlePrevious = useCallback(() => {
    if (currentIndex > 0) {
      playVideoAtIndex(currentIndex - 1);
    } else if (videoRef.current) {
      videoRef.current.currentTime = 0;
    }
  }, [currentIndex, playVideoAtIndex]);

  const handleNext = useCallback(() => {
    if (currentIndex < availableVideos.length - 1) {
      playVideoAtIndex(currentIndex + 1);
    }
  }, [currentIndex, availableVideos.length, playVideoAtIndex]);

  const handleRestart = useCallback(() => {
    if (availableVideos.length > 0) {
      playVideoAtIndex(0);
    }
  }, [availableVideos.length, playVideoAtIndex]);

  const handleReplayCurrent = useCallback(() => {
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play().catch(() => {});
    }
  }, []);

  const handleSpeedChange = useCallback((newSpeed) => {
    setPlaybackSpeed(newSpeed);
    if (videoRef.current) {
      videoRef.current.playbackRate = newSpeed;
    }
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    video.addEventListener('ended', handleVideoEnd);
    video.addEventListener('error', handleVideoError);

    return () => {
      video.removeEventListener('ended', handleVideoEnd);
      video.removeEventListener('error', handleVideoError);
    };
  }, [handleVideoEnd, handleVideoError]);

  useEffect(() => {
    if (availableVideos.length > 0 && !isPlaying && hasUserInteracted && currentIndex === 0) {
      playVideoAtIndex(0);
    }
  }, [availableVideos.length, hasUserInteracted]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
        return;
      }
      
      switch (e.key) {
        case ' ':
        case 'k':
          e.preventDefault();
          handlePlayPause();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          handlePrevious();
          break;
        case 'ArrowRight':
          e.preventDefault();
          handleNext();
          break;
        case 'r':
          if (e.shiftKey) {
            e.preventDefault();
            handleRestart();
          } else {
            e.preventDefault();
            handleReplayCurrent();
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handlePlayPause, handlePrevious, handleNext, handleRestart, handleReplayCurrent]);

  if (videos.length === 0) {
    return (
      <div className="isl-video-player empty">
        <div className="player-empty-state">
          <p>No gloss sequence to display</p>
        </div>
      </div>
    );
  }

  if (availableVideos.length === 0) {
    return (
      <div className="isl-video-player empty">
        <div className="player-empty-state">
          <h3>No video assets available</h3>
          <p>Video files for these signs have not been added yet.</p>
          <div className="gloss-sequence-display">
            {glossSequence.map((gloss, i) => (
              <span key={i} className="gloss-badge unavailable">{gloss}</span>
            ))}
          </div>
          <p className="hint">Add MP4 files to <code>backend/data/videos/</code> matching the video_file names in the dictionary.</p>
        </div>
      </div>
    );
  }

  const currentVideo = availableVideos[currentIndex];
  const currentGloss = currentVideo?.word || glossSequence[currentIndex];

  return (
    <div className="isl-video-player" onMouseEnter={() => setShowControls(true)} onMouseLeave={() => setShowControls(false)}>
      <div className="gloss-sequence-bar">
        {glossSequence.map((gloss, i) => {
          const status = getVideoStatus(i);
          return (
            <span key={i} className={`gloss-badge ${status}`}>
              {gloss}
              {status === 'playing' && <span className="play-indicator" aria-hidden="true">▶</span>}
            </span>
          );
        })}
      </div>

      <div className="video-container">
        <video
          ref={videoRef}
          className="video-element"
          playsInline
          preload="metadata"
        >
          Your browser does not support the video tag.
        </video>
        {videoError && (
          <div className="video-error-overlay">
            <p>{videoError}</p>
            <button onClick={handleNext} disabled={currentIndex >= availableVideos.length - 1}>
              Skip to Next
            </button>
          </div>
        )}
      </div>

      <div className={`player-controls ${showControls || hasUserInteracted ? 'visible' : ''}`}>
        <div className="control-group">
          <button
            onClick={handleRestart}
            aria-label="Restart sequence"
            disabled={currentIndex === 0 && !isPlaying}
            title="Restart (Shift+R)"
          >
            ↺
          </button>
          <button
            onClick={handlePrevious}
            aria-label="Previous sign"
            disabled={currentIndex === 0}
            title="Previous (←)"
          >
            ‹
          </button>
          <button
            onClick={handlePlayPause}
            aria-label={isPlaying ? 'Pause' : 'Play'}
            title="Play/Pause (Space)"
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
          <button
            onClick={handleNext}
            aria-label="Next sign"
            disabled={currentIndex >= availableVideos.length - 1}
            title="Next (→)"
          >
            ›
          </button>
          <button
            onClick={handleReplayCurrent}
            aria-label="Replay current sign"
            disabled={!isPlaying}
            title="Replay Current (R)"
          >
            ⟲
          </button>
        </div>

        <div className="control-group">
          <label htmlFor="playback-speed" className="speed-label">Speed:</label>
          <select
            id="playback-speed"
            value={playbackSpeed}
            onChange={(e) => handleSpeedChange(parseFloat(e.target.value))}
            aria-label="Playback speed"
          >
            {PLAYBACK_SPEEDS.map(speed => (
              <option key={speed} value={speed}>{speed}x</option>
            ))}
          </select>
        </div>

        <div className="control-group progress-group">
          <span className="current-gloss-label">
            {currentIndex + 1} / {availableVideos.length}: {currentGloss}
          </span>
          <div className="progress-bar" role="progressbar" aria-valuenow={Math.round(((currentIndex + 1) / availableVideos.length) * 100)} aria-valuemin={0} aria-valuemax={100}>
            <div className="progress-fill" style={{ width: `${((currentIndex + 1) / availableVideos.length) * 100}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default ISLVideoPlayer;