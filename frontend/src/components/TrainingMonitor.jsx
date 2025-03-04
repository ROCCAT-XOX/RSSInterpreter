import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { RefreshCw, Play, AlertTriangle, Save, Database, ArrowRight, Zap, BookOpen, GitBranch } from 'lucide-react';

const TrainingMonitor = () => {
  const [trainingData, setTrainingData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [trainingInProgress, setTrainingInProgress] = useState(false);
  const [trainingParams, setTrainingParams] = useState({
    episodes: 50,
    learningRate: 0.01,
    gamma: 0.99,
    dropoutRate: 0.5,
    updateFrequency: 2
  });
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [modelInfo, setModelInfo] = useState(null);
  const [feedbackStats, setFeedbackStats] = useState(null);
  const [activeTab, setActiveTab] = useState('general');
  const [useFeedback, setUseFeedback] = useState(true);

  // Mock-Trainingsdaten für Demonstrationszwecke
  const mockTrainingData = {
    episode_rewards: [
      -2, -1, 0, 1, 2, 2, 3, 4, 3, 4,
      4, 5, 5, 6, 6, 7, 7, 8, 7, 8,
      8, 9, 8, 9, 9, 9, 10, 9, 10, 10,
      10, 10, 10, 10, 10, 9, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10, 10, 10
    ],
    losses: [
      2.3, 2.1, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2, 1.15, 1.1,
      1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.72, 0.7, 0.68, 0.65,
      0.62, 0.6, 0.58, 0.55, 0.52, 0.5, 0.48, 0.46, 0.44, 0.42,
      0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31,
      0.3, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24, 0.23, 0.22, 0.21
    ],
    eval_returns: [
      0.2, 0.4, 0.6, 0.7, 0.8
    ],
    // Zeitstempel für jede Episode
    timestamps: Array.from({ length: 50 }, (_, i) => {
      const date = new Date();
      date.setMinutes(date.getMinutes() - (50 - i) * 2);
      return date.toISOString();
    })
  };

  // Lade Modell- und Feedback-Infos beim Start
  useEffect(() => {
    fetchModelStatus();
  }, []);

  // Fetch Modell-Status
  const fetchModelStatus = async () => {
    setLoading(true);
    setError(null);

    try {
      // Abfrage des Modellstatus vom Backend
      const response = await fetch('http://localhost:5000/api/model/status');

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();

      setModelInfo(data.model);
      setFeedbackStats(data.feedback);

      // Auch die Trainingsdaten abrufen
      fetchTrainingData();
    } catch (error) {
      console.error('Fehler beim Abrufen des Modellstatus:', error);
      setError('Fehler beim Laden der Modellinformationen');
      setLoading(false);

      // Bei Fehler mit Dummy-Daten fortfahren
      setModelInfo({
        exists: true,
        timestamp: new Date().toISOString(),
        size_mb: 1.2
      });

      setFeedbackStats({
        count: 5,
        classes: {
          "POSITIV": 2,
          "NEUTRAL": 1,
          "NEGATIV": 2
        }
      });

      // Trainingsdaten trotzdem abrufen
      fetchTrainingData();
    }
  };

  // Fetch Trainingsdaten
  const fetchTrainingData = async () => {
    setLoading(true);
    setError(null);

    try {
      // In einer echten Implementierung würde dies vom API abrufen
      // const response = await fetch('http://localhost:5000/api/training/history');
      // const data = await response.json();

      // Für die Demonstration verwenden wir Mock-Daten
      setTimeout(() => {
        setTrainingData(mockTrainingData);
        setLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Fehler beim Abrufen der Trainingsdaten:', error);
      setError('Fehler beim Laden der Trainingsdaten');
      setLoading(false);
    }
  };

  // Training mit oder ohne Feedback starten
  const startTraining = async () => {
    setTrainingInProgress(true);
    setError(null);
    setTrainingProgress(0);

    try {
      // Bestimme den Endpunkt basierend auf der Feedback-Einstellung
      const endpoint = useFeedback
        ? 'http://localhost:5000/api/train_with_feedback'
        : 'http://localhost:5000/api/retrain';

      console.log(`Starte Training mit ${useFeedback ? 'Feedback' : 'Standard-Daten'}...`);

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingParams),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      console.log('Trainingsergebnis:', result);

      // Simuliere den Trainingsfortschritt
      let progress = 0;
      const interval = setInterval(() => {
        progress += 2;
        setTrainingProgress(progress);

        if (progress >= 100) {
          clearInterval(interval);
          setTrainingInProgress(false);
          // Aktualisiere Modellinformationen und Trainingsdaten
          fetchModelStatus();
        }
      }, 500);
    } catch (error) {
      console.error('Fehler beim Starten des Trainings:', error);
      setError(`Fehler beim Starten des Trainings: ${error.message}`);
      setTrainingInProgress(false);
    }
  };

  // Input-Änderungen verarbeiten
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setTrainingParams(prev => ({
      ...prev,
      [name]: parseFloat(value)
    }));
  };

  // Daten für Diagramme aufbereiten
  const prepareChartData = () => {
    if (!trainingData) return [];

    return trainingData.episode_rewards.map((reward, idx) => ({
      episode: idx + 1,
      reward,
      loss: idx < trainingData.losses.length ? trainingData.losses[idx] : null,
      eval: trainingData.eval_returns[Math.floor(idx / 10)] || null,
      timestamp: trainingData.timestamps[idx]
    }));
  };

  // Feedback-Daten für Diagramm aufbereiten
  const prepareFeedbackData = () => {
    if (!feedbackStats) return [];

    return [
      { name: 'Positiv', value: feedbackStats.classes.POSITIV || 0, color: '#10B981' },
      { name: 'Neutral', value: feedbackStats.classes.NEUTRAL || 0, color: '#6B7280' },
      { name: 'Negativ', value: feedbackStats.classes.NEGATIV || 0, color: '#EF4444' }
    ];
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
          <h1 className="text-2xl font-bold text-white">Training Monitor</h1>
          <p className="text-white opacity-80">Lernfortschritt des Sentiment-Analyse-Modells</p>
        </div>

        <div className="p-6">
          {/* Tabs */}
          <div className="mb-6 border-b border-gray-200">
            <nav className="flex -mb-px">
              <button
                className={`mr-8 py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'general'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                onClick={() => setActiveTab('general')}
              >
                Allgemein
              </button>
              <button
                className={`mr-8 py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'feedback'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                onClick={() => setActiveTab('feedback')}
              >
                Feedback
              </button>
              <button
                className={`py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === 'params'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
                onClick={() => setActiveTab('params')}
              >
                Parameter
              </button>
            </nav>
          </div>

          {/* Control Actions - Always visible */}
          <div className="flex justify-between items-center mb-6">
            <button
              onClick={fetchModelStatus}
              disabled={loading || trainingInProgress}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center disabled:opacity-50"
            >
              <RefreshCw size={18} className="mr-2" />
              Aktualisieren
            </button>

            <div className="flex items-center space-x-4">
              {trainingInProgress && (
                <div className="flex items-center mr-4">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500 mr-2"></div>
                  <span className="text-sm text-gray-600">Training läuft... ({trainingProgress}%)</span>
                </div>
              )}

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="use-feedback"
                  checked={useFeedback}
                  onChange={() => setUseFeedback(!useFeedback)}
                  disabled={trainingInProgress}
                  className="mr-2 h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                />
                <label htmlFor="use-feedback" className="text-sm text-gray-700">
                  Mit Feedback trainieren
                </label>
              </div>

              <button
                onClick={startTraining}
                disabled={loading || trainingInProgress}
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center disabled:opacity-50"
              >
                <Play size={18} className="mr-2" />
                Training starten
              </button>
            </div>
          </div>

          {error && (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6 flex items-start">
              <AlertTriangle size={20} className="mr-2 mt-0.5" />
              <div>{error}</div>
            </div>
          )}

          {/* Tab Content */}
          {activeTab === 'general' && (
            <div className="space-y-8">
              {/* Model Information */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h2 className="text-lg font-semibold mb-3 text-black">Modell-Informationen</h2>
                {modelInfo ? (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-white p-4 rounded shadow">
                      <div className="text-gray-500 text-sm">Status</div>
                      <div className="font-semibold">
                        {modelInfo.exists ? (
                          <span className="text-green-600">Trainiert</span>
                        ) : (
                          <span className="text-yellow-600">Nicht trainiert</span>
                        )}
                      </div>
                    </div>
                    <div className="bg-white p-4 rounded shadow">
                      <div className="text-gray-500 text-sm">Letzte Aktualisierung</div>
                      <div className="font-semibold text-black">
                        {modelInfo.timestamp ? (
                          new Date(modelInfo.timestamp).toLocaleString()
                        ) : (
                          'Nie'
                        )}
                      </div>
                    </div>
                    <div className="bg-white p-4 rounded shadow">
                      <div className="text-gray-500 text-sm">Größe</div>
                      <div className="font-semibold text-black">
                        {modelInfo.size_mb ? (
                          `${modelInfo.size_mb} MB`
                        ) : (
                          'Unbekannt'
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-gray-500">Lade Modell-Informationen...</div>
                )}
              </div>

              {/* Training Charts */}
              {loading ? (
                <div className="flex justify-center items-center py-12">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                </div>
              ) : trainingData ? (
                <div className="space-y-6">
                  {/* Charts */}
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h2 className="text-lg font-semibold mb-3">Trainingsfortschritt</h2>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={prepareChartData()} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="episode" />
                          <YAxis yAxisId="left" />
                          <YAxis yAxisId="right" orientation="right" domain={[0, 'auto']} />
                          <Tooltip
                            formatter={(value, name) => [value, name === 'reward' ? 'Reward' : name === 'loss' ? 'Loss' : 'Evaluation']}
                            labelFormatter={(label) => `Episode ${label}`}
                          />
                          <Legend />
                          <Line yAxisId="left" type="monotone" dataKey="reward" stroke="#3B82F6" name="Reward" dot={false} />
                          <Line yAxisId="right" type="monotone" dataKey="loss" stroke="#EF4444" name="Loss" dot={false} />
                          <Line yAxisId="left" type="monotone" dataKey="eval" stroke="#10B981" name="Evaluation" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-blue-50 p-4 rounded-lg">
                      <h2 className="text-lg font-semibold mb-2">Aktueller Reward</h2>
                      <p className="text-3xl font-bold text-blue-600">
                        {trainingData.episode_rewards[trainingData.episode_rewards.length - 1]}
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        von max. {trainingData.episode_rewards.length > 0 ? Math.max(...trainingData.episode_rewards) : 0}
                      </p>
                    </div>

                    <div className="bg-red-50 p-4 rounded-lg">
                      <h2 className="text-lg font-semibold mb-2">Aktueller Loss</h2>
                      <p className="text-3xl font-bold text-red-600">
                        {trainingData.losses[trainingData.losses.length - 1].toFixed(2)}
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        Start: {trainingData.losses[0].toFixed(2)}
                      </p>
                    </div>

                    <div className="bg-green-50 p-4 rounded-lg">
                      <h2 className="text-lg font-semibold mb-2">Klassifizierung</h2>
                      <p className="text-3xl font-bold text-green-600">
                        {trainingData.eval_returns.length > 0
                          ? `${(trainingData.eval_returns[trainingData.eval_returns.length - 1] * 10).toFixed(0)}%`
                          : "N/A"}
                      </p>
                      <p className="text-sm text-gray-500 mt-1">
                        Genauigkeit im letzten Durchlauf
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-500">
                  Keine Trainingsdaten verfügbar. Starten Sie das Training oder laden Sie bestehende Daten.
                </div>
              )}
            </div>
          )}

          {/* Feedback Tab */}
          {activeTab === 'feedback' && (
            <div className="space-y-8">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h2 className="text-lg font-semibold mb-3">Feedback-Übersicht</h2>

                {feedbackStats ? (
                  <div className="space-y-4">
                    <div className="bg-white p-4 rounded shadow">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-medium text-gray-700">Gesammeltes Feedback</span>
                        <span className="bg-indigo-100 text-indigo-800 px-2 py-1 rounded-full text-xs font-bold">
                          {feedbackStats.count} Einträge
                        </span>
                      </div>

                      <div className="mt-4">
                        <h3 className="text-sm font-medium text-gray-500 mb-3">Verteilung nach Klassen</h3>
                        <div className="h-40">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={prepareFeedbackData()}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" />
                              <YAxis />
                              <Tooltip />
                              <Bar dataKey="value" name="Anzahl">
                                {prepareFeedbackData().map((entry, index) => (
                                  <rect key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>

                    <div className="bg-white p-4 rounded shadow">
                      <h3 className="font-medium text-gray-700 mb-3">Feedback-Auswirkung</h3>

                      <div className="relative pt-1">
                        <div className="mb-2 flex items-center justify-between">
                          <div>
                            <span className="text-xs font-semibold inline-block text-indigo-600">
                              Trainingsgewichtung
                            </span>
                          </div>
                          <div className="text-right">
                            <span className="text-xs font-semibold inline-block text-indigo-600">
                              3x
                            </span>
                          </div>
                        </div>
                        <div className="flex mb-2 items-center justify-between">
                          <div>
                            <span className="text-xs font-semibold inline-block py-1 px-2 uppercase rounded-full text-green-600 bg-green-200">
                              Modellgenauigkeit mit Feedback
                            </span>
                          </div>
                          <div className="text-right">
                            <span className="text-xs font-semibold inline-block text-green-600">
                              +18%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-6 text-gray-500">
                    Lade Feedback-Informationen...
                  </div>
                )}
              </div>

              <div className="bg-gray-50 p-4 rounded-lg">
                <h2 className="text-lg font-semibold mb-3">Feedback-Integration</h2>

                <div className="space-y-4">
                  <div className="bg-white p-4 rounded shadow">
                    <h3 className="font-medium text-gray-700 mb-3">Training mit Feedback</h3>
                    <p className="text-gray-600 mb-4">
                      Das Training mit Feedback verbessert die Genauigkeit des Modells, indem es Ihre manuellen Korrekturen berücksichtigt.
                      Das Feedback wird mit einer höheren Gewichtung in das Training einbezogen, um sicherzustellen, dass das Modell besonders von diesen Beispielen lernt.
                    </p>

                    <div className="flex items-center mt-2">
                      <button
                        onClick={() => {
                          setUseFeedback(true);
                          startTraining();
                        }}
                        disabled={loading || trainingInProgress || !feedbackStats || feedbackStats.count === 0}
                        className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 flex items-center disabled:opacity-50"
                      >
                        <Database size={18} className="mr-2" />
                        Mit Feedback neu trainieren
                      </button>

                      {(!feedbackStats || feedbackStats.count === 0) && (
                        <span className="ml-3 text-sm text-yellow-600">
                          Kein Feedback zum Trainieren verfügbar
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Parameters Tab */}
          {activeTab === 'params' && (
            <div className="space-y-8">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h2 className="text-lg font-semibold mb-3">Trainingsparameter</h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-4 border border-gray-200 rounded-md bg-white">
                    <h3 className="font-medium text-gray-700 mb-2">Lernrate</h3>
                    <input
                      type="range"
                      name="learningRate"
                      min="0.0001"
                      max="0.1"
                      step="0.0001"
                      value={trainingParams.learningRate}
                      onChange={handleInputChange}
                      disabled={trainingInProgress}
                      className="w-full"
                    />
                    <div className="flex justify-between text-sm text-gray-500">
                      <span>0.0001</span>
                      <span>{trainingParams.learningRate.toFixed(4)}</span>
                      <span>0.1</span>
                    </div>
                  </div>

                  <div className="p-4 border border-gray-200 rounded-md bg-white">
                    <h3 className="font-medium text-gray-700 mb-2">Gamma (Discount Faktor)</h3>
                    <input
                      type="range"
                      name="gamma"
                      min="0.9"
                      max="0.999"
                      step="0.001"
                      value={trainingParams.gamma}
                      onChange={handleInputChange}
                      disabled={trainingInProgress}
                      className="w-full"
                    />
                    <div className="flex justify-between text-sm text-gray-500">
                      <span>0.9</span>
                      <span>{trainingParams.gamma.toFixed(3)}</span>
                      <span>0.999</span>
                    </div>
                  </div>

                  <div className="p-4 border border-gray-200 rounded-md bg-white">
                    <h3 className="font-medium text-gray-700 mb-2">Dropout Rate</h3>
                    <input
                      type="range"
                      name="dropoutRate"
                      min="0.0"
                      max="0.9"
                      step="0.1"
                      value={trainingParams.dropoutRate}
                      onChange={handleInputChange}
                      disabled={trainingInProgress}
                      className="w-full"
                    />
                    <div className="flex justify-between text-sm text-gray-500">
                      <span>0.0</span>
                      <span>{trainingParams.dropoutRate.toFixed(1)}</span>
                      <span>0.9</span>
                    </div>
                  </div>

                  <div className="p-4 border border-gray-200 rounded-md bg-white">
                    <h3 className="font-medium text-gray-700 mb-2">Episoden</h3>
                    <input
                      type="number"
                      name="episodes"
                      min="10"
                      max="1000"
                      step="10"
                      value={trainingParams.episodes}
                      onChange={handleInputChange}
                      disabled={trainingInProgress}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>

                <div className="mt-4 p-4 border border-gray-200 rounded-md bg-white">
                  <h3 className="font-medium text-gray-700 mb-2">Update Frequenz</h3>
                  <div className="grid grid-cols-5 gap-2">
                    {[1, 2, 5, 10, 20].map(value => (
                      <button
                        key={value}
                        className={`py-2 px-4 border rounded-md ${
                          trainingParams.updateFrequency === value
                            ? 'bg-indigo-100 border-indigo-300 text-indigo-700'
                            : 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                        }`}
                        onClick={() => setTrainingParams(prev => ({ ...prev, updateFrequency: value }))}
                        disabled={trainingInProgress}
                      >
                        {value}
                      </button>
                    ))}
                  </div>
                  <p className="mt-2 text-xs text-gray-500">
                    Anzahl der Samples, nach denen ein Update des Modells durchgeführt wird
                  </p>
                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg">
                <h2 className="text-lg font-semibold mb-3">Parameter-Presets</h2>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <button
                    className="p-4 border border-gray-200 rounded-md bg-white hover:bg-gray-50"
                    onClick={() => setTrainingParams({
                      episodes: 50,
                      learningRate: 0.01,
                      gamma: 0.99,
                      dropoutRate: 0.5,
                      updateFrequency: 2
                    })}disabled={trainingInProgress}
                  >
                    <h3 className="font-medium text-gray-700">Standard</h3>
                    <p className="text-sm text-gray-500 mt-1">
                      Ausgeglichene Parameter für den Einstieg
                    </p>
                  </button>

                  <button
                    className="p-4 border border-gray-200 rounded-md bg-white hover:bg-gray-50"
                    onClick={() => setTrainingParams({
                      episodes: 100,
                      learningRate: 0.001,
                      gamma: 0.999,
                      dropoutRate: 0.3,
                      updateFrequency: 1
                    })}
                    disabled={trainingInProgress}
                  >
                    <h3 className="font-medium text-gray-700">Präzision</h3>
                    <p className="text-sm text-gray-500 mt-1">
                      Langsames Training mit hoher Genauigkeit
                    </p>
                  </button>

                  <button
                    className="p-4 border border-gray-200 rounded-md bg-white hover:bg-gray-50"
                    onClick={() => setTrainingParams({
                      episodes: 30,
                      learningRate: 0.02,
                      gamma: 0.95,
                      dropoutRate: 0.7,
                      updateFrequency: 5
                    })}
                    disabled={trainingInProgress}
                  >
                    <h3 className="font-medium text-gray-700">Schnell</h3>
                    <p className="text-sm text-gray-500 mt-1">
                      Schnelles Training mit weniger Genauigkeit
                    </p>
                  </button>
                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg">
                <h2 className="text-lg font-semibold mb-3">Erweiterte Optionen</h2>
                <div className="bg-yellow-50 p-4 rounded-md border border-yellow-200">
                  <div className="flex items-start">
                    <div className="flex-shrink-0">
                      <AlertTriangle size={20} className="text-yellow-500" />
                    </div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-yellow-800">Experimentelle Funktionen</h3>
                      <div className="mt-2 text-sm text-yellow-700">
                        <p>
                          Die erweiterten Optionen sind experimentell und können die Stabilität des Trainings beeinflussen.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingMonitor;