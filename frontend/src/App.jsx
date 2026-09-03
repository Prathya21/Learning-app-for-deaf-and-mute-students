import React, { useState } from 'react';
import { api, ApiError } from './services/api';
import ISLVideoPlayer from './components/ISLVideoPlayer';
import HandTracker from './components/HandTracker';
import { LanguageProvider } from './hooks/useTranslation';

function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleTranslate = async (e) => {
    e.preventDefault();
    if (!inputText.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const data = await api.translateTextToIsl(inputText);
      setResult(data);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('An unexpected error occurred');
      }
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <LanguageProvider>
      <div className="container">
        <header>
          <h1>EduSign</h1>
          <p>Breaking communication barriers through Indian Sign Language</p>
        </header>

        <div className="dashboard">
          <section className="card">
            <h2>Text to ISL</h2>
            <form onSubmit={handleTranslate}>
              <div className="form-group">
                <label htmlFor="text-input">Enter text</label>
                <textarea
                  id="text-input"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="e.g., Hello teacher, please learn"
                  rows={4}
                />
              </div>
              <button type="submit" className="btn btn-primary" disabled={loading || !inputText.trim()}>
                {loading && <span className="loading" />}
                {loading ? 'Translating...' : 'Translate'}
              </button>
            </form>

            {error && <div className="error-message">{error}</div>}

            {result && (
              <div className="form-group">
                <label>Gloss Sequence</label>
                <div className="output-area">
                  {result.gloss_sequence.length > 0
                    ? result.gloss_sequence.join(' → ')
                    : 'No gloss generated'}
                </div>
              </div>
            )}
          </section>

          <section className="card">
            <h2>ISL Video Player</h2>
            {result ? (
              <ISLVideoPlayer
                videos={result.videos}
                glossSequence={result.gloss_sequence}
              />
            ) : (
              <div className="output-area empty">Enter text and click Translate to play ISL video sequence</div>
            )}
          </section>
        </div>

        <section className="card">
          <HandTracker />
        </section>

        <section className="card">
          <h2>Future Modules</h2>
          <div className="modules">
            <div className="module-card active">
              <h3>Text to ISL</h3>
              <p>Convert text to sign language videos (Active)</p>
            </div>
            <div className="module-card">
              <h3>Live Classroom</h3>
              <p>Real-time lecture translation</p>
            </div>
            <div className="module-card">
              <h3>Gesture to Text</h3>
              <p>Webcam-based sign recognition</p>
            </div>
            <div className="module-card">
              <h3>AAC Communication</h3>
              <p>Augmentative communication board</p>
            </div>
            <div className="module-card">
              <h3>YouTube Learning</h3>
              <p>Educational video translation</p>
            </div>
          </div>
        </section>
      </div>
    </LanguageProvider>
  );
}

export default App;