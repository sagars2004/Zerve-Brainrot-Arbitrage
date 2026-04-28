# Zerve-Brainrot-Arbitrage
Real-time signal intelligence that maps the lag between pop culture virality, prediction market odds, and stock price movement to test whether internet “brainrot” predicts real market behavior.

## Inspiration

I kept noticing the same weird pattern online: a meme explodes, social feeds flood with one topic, and only later the market starts to react. It made me wonder whether internet "brainrot" could be treated like an early signal instead of just noise.
Brainrot Arbitrage Engine started from that curiosity: can we systematically measure the lag between cultural virality, prediction market sentiment, and stock movement, then turn that lag into actionable intelligence?

## What it does

Brainrot Arbitrage Engine is a real-time signal intelligence tool that tracks three streams at once:

- Virality signals from internet culture and social momentum
- Prediction market odds to capture crowd expectations
- Stock price movement to measure real market response

It aligns those timelines, detects lag windows, and highlights moments where internet sentiment appears to lead market behavior. In short, it helps answer: "Is this meme moment just hype, or an early market signal?"

## How we built it

We built the project as a data and analytics pipeline with a visual layer:

- Pulled and normalized inputs from multiple sources (social/virality, prediction markets, market prices)
- Timestamp-aligned events into a shared timeline for cross-signal comparison
- Engineered metrics to quantify divergence and lag between attention, odds, and price
- Built logic to surface candidate arbitrage windows when signals desynchronize
- Added a Python visualization layer (`visualize.py`) to make trends, lead-lag behavior, and anomalies interpretable in real time

The core idea was to move from raw trend-chasing to structured, measurable signal analysis.

## Challenges we ran into

- Noisy data: internet trends are chaotic, so separating true signal from random spikes was hard.
- Time alignment: different feeds update at different cadences and with inconsistent timestamps.
- False positives: not every viral event is financially relevant; filtering for material events took iteration.
- Cross-domain mapping: translating meme-level concepts into tickers/markets required careful heuristics.
- Interpretability: we needed visuals and metrics that made the output understandable, not just technically correct.

## Accomplishments that we're proud of

- Built an end-to-end MVP that combines culture, prediction markets, and equities in one framework
- Created a working lead-lag analysis approach instead of relying on anecdotal "it feels correlated" claims
- Delivered visual outputs that make complex multi-stream behavior readable quickly
- Defined a repeatable method for testing whether online virality has predictive value
- Turned a fun, unconventional idea into a serious experimental market intelligence tool

## What we learned

- Culture data can be useful, but only when paired with rigorous timestamping and normalization.
- Prediction markets provide a valuable bridge between social hype and financial pricing.
- Simple, transparent metrics often outperform overly complex models in early-stage signal discovery.
- The biggest challenge is not modeling, it is data quality, alignment, and reliable interpretation.
- Building fast feedback loops through visualization dramatically improved our research velocity.

## What's next for Brainrot Arbitrage Engine

- Expand signal coverage to more platforms, event types, and market instruments
- Improve entity linking between viral topics and tradable assets
- Add confidence scoring and regime detection to reduce low-quality alerts
- Backtest strategies across longer historical windows and different volatility conditions
- Move from dashboard insights to automated alerting and strategy execution workflows
