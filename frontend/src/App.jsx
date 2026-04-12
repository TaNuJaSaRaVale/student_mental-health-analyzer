import { useState } from 'react';
import { BrainCircuit, Activity, AlertCircle, CheckCircle2, TrendingUp, User, HeartPulse, Stethoscope, RefreshCw } from 'lucide-react';
import { getPrediction } from './api';

function App() {
    const [formData, setFormData] = useState({
      gender: 1, // Default Female
      age: 20,
      cgpa: 8.5,
      marital_status: 0, // Default No
      anxiety: 0,
      panic_attack: 0,
      treatment: 0
    });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    // Map string values to appropriate types
    let parsedValue = value;
    if (type === 'number' || type === 'range') {
      parsedValue = parseFloat(value);
    } else {
      parsedValue = parseInt(value, 10);
    }
    
    setFormData((prev) => ({
      ...prev,
      [name]: parsedValue
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      // The ML model was trained on a 4.0 scale dataset. 
      // We map the 10.0 scale UI input back to 4.0 scale here.
      const mappedFormData = {
        ...formData,
        cgpa: (formData.cgpa / 10.0) * 4.0
      };
      
      const data = await getPrediction(mappedFormData);
      setResult(data);
    } catch (err) {
      setError('Failed to connect to the prediction server. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setResult(null);
    setError('');
  };

  return (
    <div className="w-full" style={{ padding: '2rem 1rem' }}>
      <header className="text-center mb-8 animate-fade-in">
        <div className="flex justify-center mb-4">
          <div style={{ background: 'rgba(99, 102, 241, 0.2)', padding: '1rem', borderRadius: '50%' }}>
            <BrainCircuit size={48} color="#6366f1" />
          </div>
        </div>
        <h1 style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>Mindscape<span style={{ color: 'var(--primary)' }}>AI</span></h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '1.1rem', maxWidth: '600px', margin: '0 auto' }}>
          Advanced AI-powered student mental wellness assessment. Fast, confidential, and highly accurate.
        </p>
      </header>

      <main className="max-w-3xl animate-fade-in delay-100">
        <div className="glass-panel">
          {!result ? (
            <form onSubmit={handleSubmit}>
              <h2 className="mb-4 flex items-center gap-4" style={{ fontSize: '1.5rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '1rem' }}>
                <Activity size={24} color="#8b5cf6" /> Your Profile Assessment
              </h2>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                <div className="input-group">
                  <label className="input-label flex items-center" style={{ gap: '0.25rem' }}>
                    <User size={16} /> Gender
                  </label>
                  <select name="gender" value={formData.gender} onChange={handleInputChange} className="input-field">
                    <option value={1}>Female</option>
                    <option value={0}>Male</option>
                  </select>
                </div>

                <div className="input-group">
                  <label className="input-label flex items-center" style={{ gap: '0.25rem' }}>
                     Age: {formData.age}
                  </label>
                  <input 
                    type="range" 
                    name="age" 
                    min="16" max="30" 
                    value={formData.age} 
                    onChange={handleInputChange} 
                    style={{ width: '100%', marginTop: '0.5rem', accentColor: 'var(--primary)' }} 
                  />
                </div>

                <div className="input-group">
                  <label className="input-label flex items-center" style={{ gap: '0.25rem' }}>
                    <TrendingUp size={16} /> Latest CGPA: {formData.cgpa.toFixed(2)}
                  </label>
                  <input 
                    type="range" 
                    name="cgpa" 
                    min="0.0" max="10.0" step="0.1"
                    value={formData.cgpa} 
                    onChange={handleInputChange} 
                    style={{ width: '100%', marginTop: '0.5rem', accentColor: 'var(--primary)' }} 
                  />
                </div>

                <div className="input-group">
                  <label className="input-label">Marital Status</label>
                  <select name="marital_status" value={formData.marital_status} onChange={handleInputChange} className="input-field">
                    <option value={0}>Single</option>
                    <option value={1}>Married</option>
                  </select>
                </div>
                
                <div className="input-group">
                  <label className="input-label flex items-center" style={{ gap: '0.25rem' }}>
                    <HeartPulse size={16} color="#f59e0b" /> Experiencing anxiety?
                  </label>
                  <select name="anxiety" value={formData.anxiety} onChange={handleInputChange} className="input-field">
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </div>

                <div className="input-group">
                  <label className="input-label flex items-center" style={{ gap: '0.25rem' }}>
                    <AlertCircle size={16} color="#ef4444" /> Had panic attacks?
                  </label>
                  <select name="panic_attack" value={formData.panic_attack} onChange={handleInputChange} className="input-field">
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </div>

                <div className="input-group">
                  <label className="input-label flex items-center" style={{ gap: '0.25rem' }}>
                    <Stethoscope size={16} color="#10b981" /> Seeking specialist treatment?
                  </label>
                  <select name="treatment" value={formData.treatment} onChange={handleInputChange} className="input-field">
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </div>
              </div>

              {error && (
                <div className="mt-4 mb-4 result-box result-high animate-fade-in" style={{ padding: '1rem', marginTop: '1rem' }}>
                  <p className="flex items-center gap-4" style={{ color: 'var(--danger)', fontSize: '0.9rem' }}>
                    <AlertCircle size={20} /> {error}
                  </p>
                </div>
              )}

              <div className="mt-8 text-center">
                <button type="submit" className="btn btn-primary" disabled={loading}>
                  {loading ? (
                    <>
                      <RefreshCw className="animate-spin" size={20} style={{ animation: 'spin 1s linear infinite' }} />
                      Analyzing Health Data...
                    </>
                  ) : (
                    <>
                      <BrainCircuit size={20} /> Evaluate Mental Health Risk
                    </>
                  )}
                </button>
              </div>
            </form>
          ) : (
            <div className="animate-fade-in text-center">
              <div className="mb-8 p-4">
                <h2 style={{ fontSize: '2rem', marginBottom: '1rem' }}>Assessment Complete</h2>
                <p style={{ color: 'var(--text-muted)' }}>Based on your input, our AI model has evaluated the risk probabilities.</p>
              </div>

              <div className={`result-box ${result.has_depression ? 'result-high' : 'result-low'}`}>
                {result.has_depression ? (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
                      <AlertCircle size={64} color="var(--danger)" />
                    </div>
                    <h3 style={{ fontSize: '1.8rem', marginBottom: '0.5rem' }}>Elevated Risk Detected</h3>
                    <p style={{ marginBottom: '1rem' }}>Our model indicates signs consistent with depression. Please consider speaking with a mental health professional or counselor.</p>
                  </>
                ) : (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1rem' }}>
                      <CheckCircle2 size={64} color="var(--success)" />
                    </div>
                    <h3 style={{ fontSize: '1.8rem', marginBottom: '0.5rem' }}>Low Risk Detected</h3>
                    <p style={{ marginBottom: '1rem' }}>Our model indicates your current symptoms represent a lower baseline risk. Always prioritize self-care.</p>
                  </>
                )}

                <div style={{ marginTop: '2rem', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px' }}>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>Model Probability Score</p>
                  <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', marginTop: '0.5rem', overflow: 'hidden' }}>
                    <div 
                      style={{ 
                        height: '100%', 
                        width: `${(result.probability * 100).toFixed(1)}%`, 
                        background: result.has_depression ? 'var(--danger)' : 'var(--success)',
                        transition: 'width 1s ease-out'
                      }} 
                    />
                  </div>
                  <p style={{ marginTop: '0.5rem', fontWeight: 'bold' }}>
                    {(result.probability * 100).toFixed(1)}% Risk Likelihood
                  </p>
                </div>
              </div>

              <div className="mt-8">
                <button onClick={resetForm} className="btn" style={{ background: 'transparent', border: '1px solid var(--glass-border)', color: 'var(--text-main)' }}>
                  <RefreshCw size={18} /> Start New Assessment
                </button>
              </div>
            </div>
          )}
        </div>
      </main>
      
      <style>{`
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

export default App;
