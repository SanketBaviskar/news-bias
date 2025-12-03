import { useState } from "react";
import axios from "axios";
import { AlertTriangle, CheckCircle, Loader2 } from "lucide-react";

function App() {
	const [text, setText] = useState("");
	const [loading, setLoading] = useState(false);
	const [result, setResult] = useState(null);
	const [error, setError] = useState(null);

	const handleSubmit = async (e) => {
		e.preventDefault();
		setLoading(true);
		setError(null);
		setResult(null);

		try {
			const response = await axios.post(
				import.meta.env.VITE_API_URL || "http://localhost:5000/predict",
				{ news_text: text },
				{ headers: { "Content-Type": "multipart/form-data" } }
			);
			setResult(response.data);
		} catch (err) {
			console.error(err);
			setError(
				"An error occurred while analyzing the text. Please try again."
			);
		} finally {
			setLoading(false);
		}
	};

	const isBiased =
		result &&
		result.prediction_text.includes("biased") &&
		!result.prediction_text.includes("not biased");

	return (
		<div className="container">
			<h1>News Bias Detection</h1>

			<form onSubmit={handleSubmit}>
				<div>
					<label htmlFor="news_text">Enter News Article Text:</label>
					<textarea
						id="news_text"
						value={text}
						onChange={(e) => setText(e.target.value)}
						placeholder="Paste the news article content here..."
						required
					/>
				</div>

				<button type="submit" disabled={loading || !text.trim()}>
					{loading ? (
						<>
							<Loader2 className="spinner" />
							<span>Analyzing...</span>
						</>
					) : (
						<span>Analyze Text</span>
					)}
				</button>
			</form>

			{error && (
				<div
					className="result-card"
					style={{
						borderColor: "var(--danger)",
						color: "var(--danger)",
					}}
				>
					<div className="result-header">
						<AlertTriangle />
						Error
					</div>
					<p>{error}</p>
				</div>
			)}

			{result && (
				<div className="result-card">
					<div
						className={`result-header ${
							isBiased ? "biased-header" : "unbiased-header"
						}`}
					>
						{isBiased ? <AlertTriangle /> : <CheckCircle />}
						{result.prediction_text}
					</div>

					<div className="explanation">
						<strong>Analysis:</strong> {result.explanation}
					</div>

					<div className="confidence-section">
						<div className="confidence-label">
							<span>Confidence Score</span>
							<span>{(result.confidence * 100).toFixed(1)}%</span>
						</div>
						<div className="progress-bar-bg">
							<div
								className={`progress-bar-fill ${
									isBiased
										? "progress-biased"
										: "progress-unbiased"
								}`}
								style={{ width: `${result.confidence * 100}%` }}
							></div>
						</div>
					</div>
				</div>
			)}
		</div>
	);
}

export default App;
