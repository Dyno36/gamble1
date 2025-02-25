import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.title("Player Prop Betting Simulator")
st.write("Simulate player prop outcomes with Monte Carlo and Bayesian updating to find value bets!")

# Sidebar inputs for manual entry
st.sidebar.header("Player Stats Input")
player_name = st.sidebar.text_input("Player Name", "Example Player")
player_position = st.sidebar.selectbox("Player Position", ["PG", "SG", "SF", "PF", "C"])
mean_points = st.sidebar.number_input("Average Points per Game", value=20.0)
std_dev_points = st.sidebar.number_input("Standard Deviation (Points)", value=5.0)
games_played = st.sidebar.number_input("Number of Games Played", value=30)

# Recent performance for Bayesian updating
st.sidebar.header("Recent Performance")
recent_avg_points = st.sidebar.number_input("Recent Average Points", value=22.0)
recent_games = st.sidebar.number_input("Number of Recent Games", value=5)

# Opponent defense stats by position
st.sidebar.header("Opponent Defense Stats (Points Allowed by Position)")
opp_points_allowed_position = st.sidebar.number_input(f"Opponent Points Allowed to {player_position}", value=22.0)
league_avg_points_allowed_position = st.sidebar.number_input(f"League Average Points Allowed to {player_position}", value=24.0)

# Sportsbook line and odds
st.sidebar.header("Bet Details")
line = st.sidebar.number_input("Sportsbook Line", value=20.5)
odds = st.sidebar.number_input("Bet Odds (e.g., -110 for American odds)", value=-110)
simulations = st.sidebar.slider("Number of Monte Carlo Simulations", 1000, 20000, 10000)

# Bayesian updating function
def bayesian_update(prior_mu, prior_sigma, recent_mu, recent_games):
    posterior_mu = (prior_mu / prior_sigma**2 + recent_games * recent_mu / prior_sigma**2) / (1 / prior_sigma**2 + recent_games / prior_sigma**2)
    posterior_sigma = np.sqrt(1 / (1 / prior_sigma**2 + recent_games / prior_sigma**2))
    return posterior_mu, posterior_sigma

# Adjust points based on opponent defense by position
def adjust_for_opponent_defense(base_points, opp_points_allowed, league_avg_points_allowed):
    defense_factor = opp_points_allowed / league_avg_points_allowed
    adjusted_points = base_points * defense_factor
    return adjusted_points

# Monte Carlo simulation
def monte_carlo_simulation(mu, sigma, sims):
    return np.random.normal(mu, sigma, sims)

# EV calculation
def calculate_ev(prob_over, odds):
    if odds < 0:
        odds_decimal = 1 + (100 / abs(odds))
    else:
        odds_decimal = (odds / 100) + 1
    ev = (prob_over * odds_decimal) - (1 - prob_over)
    return ev

# Calculate edge vs sportsbook line
def calculate_edge(projected_points, line):
    return ((projected_points - line) / line) * 100

# Perform Bayesian update
posterior_mu, posterior_sigma = bayesian_update(mean_points, std_dev_points, recent_avg_points, recent_games)

# Adjust for opponent defense by position
adjusted_mu = adjust_for_opponent_defense(posterior_mu, opp_points_allowed_position, league_avg_points_allowed_position)

# Run Monte Carlo simulation
simulated_points = monte_carlo_simulation(adjusted_mu, posterior_sigma, simulations)

# Calculate probabilities
prob_over_line = np.mean(simulated_points > line)
ev = calculate_ev(prob_over_line, odds)

# Calculate edge percentage
edge_percentage = calculate_edge(adjusted_mu, line)

# Results
st.subheader(f"Results for {player_name} ({player_position})")
st.write(f"**Updated Mean (Posterior):** {posterior_mu:.2f}")
st.write(f"**Adjusted Mean (vs Opponent for {player_position} — Projected Points):** {adjusted_mu:.2f}")
st.write(f"**Updated Standard Deviation:** {posterior_sigma:.2f}")
st.write(f"**Probability of Hitting Over {line} Points:** {prob_over_line * 100:.2f}%")
st.write(f"**Expected Value (EV):** {ev:.2f}")
st.write(f"**Edge vs Sportsbook Line:** {edge_percentage:.2f}%")

# Plot the distribution
st.subheader("Simulated Outcome Distribution")
fig, ax = plt.subplots()
ax.hist(simulated_points, bins=30, color="blue", alpha=0.6, edgecolor="black")
ax.axvline(line, color="red", linestyle="--", label=f"Sportsbook Line: {line}")
ax.axvline(adjusted_mu, color="green", linestyle="--", label=f"Projected Points: {adjusted_mu:.2f}")
ax.set_title("Monte Carlo Simulated Player Points")
ax.set_xlabel("Points")
ax.set_ylabel("Frequency")
ax.legend()
st.pyplot(fig)

# Bet recommendation
if ev > 0:
    st.success("Positive EV! This bet could be profitable long-term.")
else:
    st.warning("Negative EV. This bet may not have long-term value.")