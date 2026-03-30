import random

def explain_prediction(change):
    reasons_up = [
        "📈 Strong buying trend",
        "🔥 Positive market sentiment",
        "💰 High investor confidence",
        "🚀 Recent news impact"
    ]

    reasons_down = [
        "📉 Selling pressure",
        "⚠️ Market uncertainty",
        "💸 Profit booking",
        "🌍 Global factors"
    ]

    if change > 0:
        return random.choice(reasons_up)
    else:
        return random.choice(reasons_down)