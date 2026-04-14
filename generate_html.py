"""
Generate HTML page from value bets CSV
Updates index.html with latest predictions
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
import glob

# Find latest value bets CSV
data_dir = Path("data")
csv_files = glob.glob(str(data_dir / "value_bets_*.csv"))

if not csv_files:
    print("No value bets CSV found. Run predictions first:")
    print("python main.py --predict --odds --leagues PL --days 7")
    exit(1)

latest_csv = max(csv_files)
print(f"Loading predictions from: {latest_csv}")

# Load data
df = pd.read_csv(latest_csv)
print(f"Found {len(df)} value bets")

# Calculate stats safely to avoid ZeroDivisionError
total_bets = len(df)
if total_bets > 0:
    avg_confidence = df['confidence'].mean() * 100
    avg_edge = df['edge'].mean() * 100
    total_stake = df['kelly_stake_pct'].sum()
    potential_roi = (df['expected_value'].sum() / total_bets) * 100
else:
    avg_confidence = 0
    avg_edge = 0
    total_stake = 0
    potential_roi = 0

# Generate HTML
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Football Predictions - Value Bets Today</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .update-time {{
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 10px;
        }}
        
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .stat-card .number {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .bet-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            transition: transform 0.2s;
        }}
        
        .bet-card:hover {{
            transform: translateY(-5px);
        }}
        
        .bet-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        
        .match-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
        }}
        
        .match-time {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .confidence-badge {{
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }}
        
        .confidence-high {{
            background: #10b981;
            color: white;
        }}
        
        .confidence-medium {{
            background: #f59e0b;
            color: white;
        }}
        
        .prediction-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
        }}
        
        .prediction-value {{
            font-size: 1.5em;
            font-weight: bold;
        }}
        
        .edge-indicator {{
            background: rgba(255,255,255,0.3);
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
        
        .bet-details {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 15px;
        }}
        
        .detail-item {{
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .detail-label {{
            font-size: 0.8em;
            color: #666;
        }}
        
        .detail-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
        
        .profit-estimate {{
            background: #ecfdf5;
            border: 2px solid #10b981;
            padding: 15px;
            border-radius: 10px;
        }}
        
        .profit-estimate h4 {{
            color: #059669;
            margin-bottom: 10px;
        }}
        
        .profit-row {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
        }}
        
        .visitor-count {{
            background: rgba(255,255,255,0.2);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            display: inline-block;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        
        @media (max-width: 768px) {{
            .header h1 {{
                font-size: 1.8em;
            }}
            .bet-details {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚽ 次级联赛的价值投注推荐</h1>
            <div class="subtitle">AI-Powered Predictions with Proven Edge</div>
            <div class="update-time">📅 Updated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
            <div class="visitor-count">👥 <span id="visitor-count">Loading...</span></div>
        </div>
        
        <div class="stats-bar">
            <div class="stat-card">
                <div class="number">{total_bets}</div>
                <div class="label">Value Bets Today</div>
            </div>
            <div class="stat-card">
                <div class="number">{avg_confidence:.0f}%</div>
                <div class="label">Avg Confidence</div>
            </div>
            <div class="stat-card">
                <div class="number">+{avg_edge:.0f}%</div>
                <div class="label">Average Edge</div>
            </div>
            <div class="stat-card">
                <div class="number">{potential_roi:.0f}%</div>
                <div class="label">Potential ROI</div>
            </div>
        </div>
"""

# Generate bet cards
for idx, bet in df.iterrows():
    # Confidence badge
    confidence_pct = bet['confidence'] * 100
    confidence_class = 'confidence-high' if confidence_pct >= 60 else 'confidence-medium'
    
    # Format date
    date_str = pd.to_datetime(bet['date']).strftime('%A, %b %d @ %I:%M %p GMT')
    
    # Calculate profit examples
    stake_small = int(bet['kelly_stake_pct'] / 2)
    stake_rec = int(bet['kelly_stake_pct'])
    odds = bet['bookmaker_odds']
    
    win_small = stake_small * odds
    profit_small = win_small - stake_small
    win_rec = stake_rec * odds
    profit_rec = win_rec - stake_rec
    
    # Emoji for outcome
    outcome_emoji = "🏆" if confidence_pct >= 70 else "⚽"
    
    html += f"""
        <div class="bet-card">
            <div class="bet-header">
                <div class="match-info">
                    <div class="match-title">{outcome_emoji} {bet['home_team']} vs {bet['away_team']}</div>
                    <div class="match-time">📅 {date_str}</div>
                </div>
                <div class="confidence-badge {confidence_class}">{confidence_pct:.0f}% Confidence</div>
            </div>
            
            <div class="prediction-box">
                <div class="prediction-label">🎯 PREDICTION</div>
                <div class="prediction-value">{bet['outcome']} <span class="edge-indicator">+{bet['edge']*100:.0f}% Edge</span></div>
            </div>
            
            <div class="bet-details">
                <div class="detail-item">
                    <div class="detail-label">Bookmaker Odds</div>
                    <div class="detail-value">{odds:.2f}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Our Probability</div>
                    <div class="detail-value">{bet['model_probability']*100:.1f}%</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Recommended Stake</div>
                    <div class="detail-value">${stake_rec}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Expected Value</div>
                    <div class="detail-value">+{bet['expected_value']:.2f}</div>
                </div>
            </div>
            
            <div class="profit-estimate">
                <h4>💰 Profit Estimate (100 bankroll)</h4>
                <div class="profit-row">
                    <span>Bet ${stake_small} →</span>
                    <span><strong>Win ${win_small:.2f}</strong> (+${profit_small:.2f} profit)</span>
                </div>
                <div class="profit-row">
                    <span>Bet ${stake_rec} →</span>
                    <span><strong>Win ${win_rec:.2f}</strong> (+${profit_rec:.2f} profit)</span>
                </div>
            </div>
        </div>
"""

# Footer and clean up
html += f"""
    </div>
    
    <script>
        // Simple visitor counter using localStorage
        function updateVisitorCount() {{
            const today = new Date().toDateString();
            const visitorKey = 'visitor_' + today;
            let count = parseInt(localStorage.getItem(visitorKey) || '0');
            const userVisitKey = 'user_visited_' + today;
            if (!sessionStorage.getItem(userVisitKey)) {{
                count++;
                localStorage.setItem(visitorKey, count.toString());
                sessionStorage.setItem(userVisitKey, 'true');
            }}
            document.getElementById('visitor-count').textContent = count + (count === 1 ? ' visitor today' : ' visitors today');
        }}
        
        document.addEventListener('DOMContentLoaded', function() {{
            updateVisitorCount();
        }});
    </script>
</body>
</html>
"""

# Save HTML
with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"✓ Generated index.html with {total_bets} predictions")
print(f"  Average confidence: {avg_confidence:.1f}%")
print(f"  Average edge: {avg_edge:.1f}%")
print(f"\nTo publish to GitHub Pages:")
print("  git add index.html")
print("  git commit -m 'Update predictions'")
print("  git push")
