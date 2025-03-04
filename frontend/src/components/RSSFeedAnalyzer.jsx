import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, RefreshCw, Save, ThumbsUp, ThumbsDown, ArrowRightCircle, AlertTriangle } from 'lucide-react';

const SentimentEmoji = ({ sentiment }) => {
  switch (sentiment) {
    case 'POSITIV':
      return <span className="text-green-500 text-xl">😃</span>;
    case 'NEUTRAL':
      return <span className="text-gray-500 text-xl">😐</span>;
    case 'NEGATIV':
      return <span className="text-red-500 text-xl">😟</span>;
    default:
      return null;
  }
};

const RSSFeedAnalyzer = () => {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [feedUrl, setFeedUrl] = useState('https://de.cointelegraph.com/rss/tag/bitcoin');
  const [lookbackDays, setLookbackDays] = useState(2);
  const [expandedArticle, setExpandedArticle] = useState(null);
  const [userFeedback, setUserFeedback] = useState({});
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);
  const [stats, setStats] = useState({ POSITIV: 0, NEUTRAL: 0, NEGATIV: 0 });
  const [availableFeeds, setAvailableFeeds] = useState([
    'https://de.cointelegraph.com/rss/tag/bitcoin',
    'https://de.cointelegraph.com/rss/tag/ethereum',
    'https://de.cointelegraph.com/rss/tag/litecoin',
    'https://de.cointelegraph.com/rss/tag/monero'
  ]);
  const [useMockData, setUseMockData] = useState(false);

  // Funktion zum Abrufen von Artikeln vom Backend
  const fetchArticles = async () => {
    setLoading(true);
    setError(null);

    try {
      // Bei API-Problemen kann auf Mock-Daten umgeschaltet werden
      const apiUrl = useMockData
        ? 'http://localhost:5000/api/mock_articles'
        : `http://localhost:5000/api/articles?url=${encodeURIComponent(feedUrl)}&days=${lookbackDays}`;

      console.log("Versuche Artikel abzurufen von:", apiUrl);

      const response = await fetch(apiUrl);

      if (!response.ok) {
        throw new Error(`HTTP Fehler! Status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Erhaltene Daten:", data);

      if (data.error) {
        throw new Error(data.error);
      }

      if (data.articles) {
        setArticles(data.articles);
        setStats(data.stats || { POSITIV: 0, NEUTRAL: 0, NEGATIV: 0 });
      } else {
        throw new Error("Keine Artikel in der Antwort gefunden");
      }
    } catch (error) {
      console.error('Fehler beim Abrufen der Artikel:', error);
      setError(`Fehler beim Laden der Artikel: ${error.message}. ${useMockData ? 'Versuche es später erneut.' : 'Wechsle zu Mock-Daten.'}`);

      if (!useMockData) {
        setUseMockData(true); // Bei Fehler zu Mock-Daten wechseln
        fetchMockArticles();
      }
    } finally {
      setLoading(false);
    }
  };

  // Funktion zum Abrufen von Mock-Artikeln
  const fetchMockArticles = () => {
    setLoading(true);

    // Einfache Mock-Daten, falls API nicht erreichbar
    const mockArticles = [
      {
        id: 1,
        title: 'Bitcoin erreicht neues Allzeithoch',
        date: new Date().toISOString(),
        content: 'Der Bitcoin-Kurs ist heute auf ein neues Allzeithoch von 70.000 USD gestiegen, was auf ein gestiegenes institutionelles Interesse und die Zulassung neuer ETFs zurückzuführen ist.',
        url: 'https://example.com/news/1',
        sentiment: 'POSITIV',
        confidence: 0.92,
        summary: 'Der Bitcoin-Kurs ist heute auf ein neues Allzeithoch gestiegen.'
      },
      {
        id: 2,
        title: 'Ethereum-Entwickler geben Update zum Merge',
        date: new Date(Date.now() - 86400000).toISOString(),
        content: 'Die Ethereum-Foundation hat ein Update zum Status des Merge-Upgrades gegeben. Der Übergang zu Proof of Stake verläuft wie geplant.',
        url: 'https://example.com/news/2',
        sentiment: 'NEUTRAL',
        confidence: 0.78,
        summary: 'Die Ethereum-Foundation gab ein Update zum Merge-Upgrade.'
      },
      {
        id: 3,
        title: 'Marktkorrektur: Kryptowährungen verlieren 10% an Wert',
        date: new Date(Date.now() - 2 * 86400000).toISOString(),
        content: 'In einer plötzlichen Marktkorrektur haben die meisten Kryptowährungen etwa 10% ihres Wertes verloren. Analysten führen dies auf Gewinnmitnahmen und makroökonomische Faktoren zurück.',
        url: 'https://example.com/news/3',
        sentiment: 'NEGATIV',
        confidence: 0.85,
        summary: 'Kryptowährungen verlieren 10% durch Marktkorrektur.'
      }
    ];

    setArticles(mockArticles);
    setStats({
      POSITIV: 1,
      NEUTRAL: 1,
      NEGATIV: 1
    });

    setLoading(false);
  };

  // Funktion zum Übermitteln von Feedback an das Backend
  const submitFeedback = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          feedback: userFeedback
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP Fehler! Status: ${response.status}`);
      }

      const result = await response.json();

      if (result.error) {
        throw new Error(result.error);
      }

      console.log('Feedback-Übermittlungsergebnis:', result);

      // Aktualisiere Statistiken basierend auf Benutzerfeedback
      const newStats = { ...stats };
      Object.entries(userFeedback).forEach(([id, newSentiment]) => {
        const article = articles.find(a => a.id.toString() === id);
        if (article && article.sentiment !== newSentiment) {
          newStats[article.sentiment]--;
          newStats[newSentiment]++;
        }
      });
      setStats(newStats);

      setFeedbackSubmitted(true);
    } catch (error) {
      console.error('Fehler beim Übermitteln des Feedbacks:', error);
      setError(`Fehler beim Übermitteln des Feedbacks: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Hole verfügbare Feeds vom Backend
  useEffect(() => {
    async function fetchFeeds() {
      try {
        console.log("Versuche Feeds abzurufen...");
        const response = await fetch('http://localhost:5000/api/feeds');

        if (response.ok) {
          const data = await response.json();
          console.log("Erhaltene Feeds:", data);

          if (data.feeds && data.feeds.length > 0) {
            setAvailableFeeds(data.feeds);
            setFeedUrl(data.feeds[0]);
          }
        } else {
          console.error("Fehler beim Abrufen der Feeds: HTTP Status", response.status);
        }
      } catch (error) {
        console.error('Fehler beim Abrufen der Feeds:', error);
        // Behalte die Standard-Feeds bei, wenn ein Fehler auftritt
      }
    }

    fetchFeeds();
  }, []);

  // Lade Artikel, wenn sich die Feed-URL oder der Zeitraum ändert
  useEffect(() => {
    fetchArticles();
  }, [feedUrl, lookbackDays]);

  const toggleArticle = (id) => {
    setExpandedArticle(expandedArticle === id ? null : id);
  };

  const handleFeedbackChange = (id, sentiment) => {
    setUserFeedback(prev => ({
      ...prev,
      [id]: sentiment
    }));
  };

  const handleRefresh = () => {
    setUserFeedback({});
    setFeedbackSubmitted(false);
    setUseMockData(false); // Versuche es mit echten Daten
    fetchArticles();
  };

  const toggleDataSource = () => {
    setUseMockData(!useMockData);
    setError(null);
    setTimeout(fetchArticles, 100);
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
          <h1 className="text-2xl font-bold text-white">RSS Sentiment Analyzer</h1>
          <p className="text-white opacity-80">Analyse und Bewertung von RSS-Feeds mit KI</p>
        </div>

        <div className="p-6">
          {/* Controls */}
          <div className="mb-6 space-y-4">
            <div className="flex flex-col md:flex-row md:items-center gap-4">
              <div className="flex-1">
                <label className="block text-sm font-medium text-gray-700 mb-1">RSS-Feed URL</label>
                <select
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={feedUrl}
                  onChange={(e) => setFeedUrl(e.target.value)}
                  disabled={useMockData}
                >
                  {availableFeeds.map((feed, index) => (
                    <option key={index} value={feed}>
                      {feed.split('/').slice(-2).join('/')}
                    </option>
                  ))}
                </select>
              </div>
              <div className="w-32">
                <label className="block text-sm font-medium text-gray-700 mb-1">Zeitraum (Tage)</label>
                <input
                  type="number"
                  min="1"
                  max="365"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={lookbackDays}
                  onChange={(e) => setLookbackDays(parseInt(e.target.value, 10))}
                  disabled={useMockData}
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={handleRefresh}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
                >
                  <RefreshCw size={18} className="mr-2" />
                  Aktualisieren
                </button>
              </div>
            </div>

            <div className="flex justify-between items-center">
              <div className="text-sm text-gray-500">
                {useMockData ? (
                  <span className="bg-yellow-100 text-yellow-800 px-2 py-1 rounded-md">
                    Testdaten werden angezeigt
                  </span>
                ) : (
                  <span>Live-Daten werden angezeigt</span>
                )}
              </div>
              <button
                onClick={toggleDataSource}
                className="text-sm text-blue-600 hover:underline"
              >
                {useMockData ? "Zu Live-Daten wechseln" : "Zu Testdaten wechseln"}
              </button>
            </div>
          </div>

          {/* Fehleranzeige */}
          {error && (
            <div className="mb-6 bg-red-50 border-l-4 border-red-500 p-4 flex items-start">
              <AlertTriangle size={20} className="text-red-500 mr-2 flex-shrink-0 mt-0.5" />
              <div className="text-red-700">{error}</div>
            </div>
          )}

          {/* Stats */}
          <div className="mb-6 bg-gray-50 p-4 rounded-md">
            <h2 className="text-lg font-semibold mb-3">Stimmungsanalyse</h2>
            <div className="flex flex-wrap gap-4">
              <div className="flex items-center bg-green-100 px-4 py-2 rounded-md">
                <SentimentEmoji sentiment="POSITIV" />
                <span className="ml-2 font-medium text-black">Positiv: {stats.POSITIV}</span>
              </div>
              <div className="flex items-center bg-gray-100 px-4 py-2 rounded-md">
                <SentimentEmoji sentiment="NEUTRAL" />
                <span className="ml-2 font-medium text-black">Neutral: {stats.NEUTRAL}</span>
              </div>
              <div className="flex items-center bg-red-100 px-4 py-2 rounded-md">
                <SentimentEmoji sentiment="NEGATIV" />
                <span className="ml-2 font-medium text-black">Negativ: {stats.NEGATIV}</span>
              </div>
            </div>
          </div>

          {/* Articles */}
          {loading ? (
            <div className="flex justify-center items-center py-12">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
          ) : articles.length === 0 ? (
            <div className="py-12 text-center text-gray-500">
              Keine Artikel gefunden. Versuche es mit einem anderen Feed oder Zeitraum.
            </div>
          ) : (
            <div className="space-y-4">
              {articles.map((article) => {
                const currentSentiment = userFeedback[article.id] || article.sentiment;
                const sentimentColor =
                  currentSentiment === 'POSITIV' ? 'bg-green-100 border-green-300' :
                  currentSentiment === 'NEGATIV' ? 'bg-red-100 border-red-300' :
                  'bg-gray-100 border-gray-300';

                return (
                  <div
                    key={article.id}
                    className={`border ${sentimentColor} rounded-md overflow-hidden`}
                  >
                    <div
                      className="flex justify-between items-center px-4 py-3 cursor-pointer"
                      onClick={() => toggleArticle(article.id)}
                    >
                      <div className="flex-1">
                        <div className="flex items-center">
                          <SentimentEmoji sentiment={currentSentiment} />
                          <h3 className="font-semibold ml-2 text-black">{article.title}</h3>
                        </div>
                        <p className="text-sm text-gray-500">
                          {new Date(article.date).toLocaleDateString()} -
                          Konfidenz: {Math.round(article.confidence * 100)}%
                        </p>
                      </div>
                      {expandedArticle === article.id ?
                        <ChevronUp size={20} className="text-gray-500" /> :
                        <ChevronDown size={20} className="text-gray-500" />
                      }
                    </div>

                    {expandedArticle === article.id && (
                      <div className="px-4 py-3 border-t border-gray-200">
                        <div className="mb-4">
                          <h4 className="font-medium text-gray-700 mb-1">Original:</h4>
                          <p className="text-gray-600">{article.content}</p>
                        </div>

                        <div className="mb-4">
                          <h4 className="font-medium text-gray-700 mb-1">KI-Zusammenfassung:</h4>
                          <p className="text-gray-600">{article.summary}</p>
                        </div>

                        <div className="border-t border-gray-200 pt-3 mt-4">
                          <h4 className="font-medium text-gray-700 mb-2">Stimmungseinschätzung</h4>
                          <div className="flex space-x-2">
                            <button
                              className={`text-black px-3 py-1.5 rounded-md flex items-center text-sm ${
                                currentSentiment === 'POSITIV' 
                                  ? 'bg-green-500 text-white' 
                                  : 'bg-gray-100 hover:bg-gray-200'
                              }`}
                              onClick={() => handleFeedbackChange(article.id, 'POSITIV')}
                            >
                              <ThumbsUp size={16} className="mr-1" />
                              Positiv
                            </button>
                            <button
                              className={`text-black px-3 py-1.5 rounded-md flex items-center text-sm ${
                                currentSentiment === 'NEUTRAL' 
                                  ? 'bg-gray-500 text-white' 
                                  : 'bg-gray-100 hover:bg-gray-200'
                              }`}
                              onClick={() => handleFeedbackChange(article.id, 'NEUTRAL')}
                            >
                              <ArrowRightCircle size={16} className="mr-1" />
                              Neutral
                            </button>
                            <button
                              className={`text-black px-3 py-1.5 rounded-md flex items-center text-sm ${
                                currentSentiment === 'NEGATIV' 
                                  ? 'bg-red-500 text-white' 
                                  : 'bg-gray-100 hover:bg-gray-200'
                              }`}
                              onClick={() => handleFeedbackChange(article.id, 'NEGATIV')}
                            >
                              <ThumbsDown size={16} className="mr-1" />
                              Negativ
                            </button>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}

              {/* Feedback-Übermittlung */}
              {Object.keys(userFeedback).length > 0 && (
                <div className="mt-6 flex justify-center">
                  <button
                    className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center disabled:opacity-50"
                    onClick={submitFeedback}
                    disabled={loading || feedbackSubmitted}
                  >
                    <Save size={18} className="mr-2" />
                    {feedbackSubmitted ? 'Feedback übermittelt!' : 'Feedback übermitteln'}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RSSFeedAnalyzer;