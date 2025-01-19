#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 18:21:54 2025

@author: johnnynienstedt
"""

#
# Times Thru Season Penalty
#
# Johnny Nienstedt
#

#
# Do hitters gain an advantage seeing pitchers many times over the course of a
# season, or do they forget their precious knowledge in the intervening weeks?
#

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import FormatStrFormatter
from sklearn.linear_model import LinearRegression



'''
###############################################################################
############################### Data Processing ###############################
###############################################################################
'''

# import data
data = pd.read_csv('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Data/classified_pitch_data.csv')

# sort pitches chronologically
sort_cols = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number']
sorted_df = data.sort_values(
    by=sort_cols,
    ascending=True
)

# identify pitcher-batter matchups
def add_matchup_counter(data, group='game'):

    if group not in ['game', 'season', 'total']:
        print('\nOptions for group are: "game" "season" or "total"')
        return

    colname = 'matchup_' + group    

    df = data.copy()
    pa_df = df.groupby(['game_year', 'game_date', 'game_pk', 'at_bat_number']).agg({
        'batter': 'first',
        'pitcher': 'first'
    }).reset_index()
    
    # Calculate the matchup count
    if group == 'game':
        pa_df[colname] = pa_df.groupby(['game_date', 'game_pk', 'batter', 'pitcher']).cumcount() + 1
    elif group == 'season':
        pa_df[colname] = pa_df.groupby(['game_year', 'batter', 'pitcher']).cumcount() + 1
    elif group == 'total':
        pa_df[colname] = pa_df.groupby(['batter', 'pitcher']).cumcount() + 1
    
    game_df = df.groupby(['game_year', 'game_date', 'game_pk', 'batter', 'pitcher']).size().reset_index(name='temp')
    game_df['matchup_game_number'] = game_df.groupby(['batter', 'pitcher']).cumcount() + 1
    game_df = game_df.drop('temp', axis=1)
    
    # Merge the matchup counts back to the original dataframe
    temp = df.merge(
        pa_df[['game_date', 'game_pk', 'at_bat_number', colname]],
        on=['game_date', 'game_pk', 'at_bat_number'],
        how='left'
    )
    
    result = temp.merge(
        game_df[['game_date', 'game_pk', 'batter', 'pitcher', 'matchup_game_number']],
        on = ['game_date', 'game_pk', 'batter', 'pitcher'],
        how='left')
    
    return result

# count matchups per game, season, and all time
matchup_df = add_matchup_counter(sorted_df, group = 'game')
matchup_df = add_matchup_counter(matchup_df, group = 'season')
matchup_df = add_matchup_counter(matchup_df, group = 'total')



'''
###############################################################################
################################ Data Analysis ################################
###############################################################################
'''

# weighted linear regression
def weighted_linear_regression(x, y, weights):

    # Reshape x if it's 1D
    X = np.array(x).reshape(-1, 1)
    y = np.array(y)
    weights = np.array(weights)
    
    # Create and fit the model with sample weights
    model = LinearRegression()
    model.fit(X, y, sample_weight=weights)
    
    # Calculate R-squared
    y_pred = model.predict(X)
    ss_tot = np.sum(weights * (y - np.mean(y))**2)
    ss_res = np.sum(weights * (y - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return model.coef_[0], model.intercept_, r_squared

def calculate_penalty(data, group, plot=True, trendline=False, adjusted=False):
    
    if adjusted:
        df = data.groupby('matchup_' + group).mean(numeric_only=True)['woba_adj_tto']
        adj = ' (Adjusted for TTO)'
    else:
        df = data.groupby('matchup_' + group).mean(numeric_only=True)['woba_value']
        adj = ''
    
    weight_df = data.groupby('matchup_' + group).size()
    
    x = df.index
    y = df.values
    weights = weight_df.values
    
    # regression
    slope, intercept, r_squared = weighted_linear_regression(x, y, weights)

    # plot
    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))    
        x_line = np.array([min(x), max(x)])
        y_line = slope * x_line + intercept
        
        if trendline:
            plt.plot(x_line, y_line, 'k--')
        
        # points with weight
        max_size = 1500
        scale = weights[0]/max_size
        scaled_sizes = weights/scale
        plt.scatter(x, y, s=scaled_sizes, alpha=1)
    
        # formatting
        plt.xlabel('Times Facing Batter in ' + group.title())
        plt.ylabel('wOBA')
        
        if group == 'game':
            s = 'the Order'
        if group == 'season':
            s = 'the Season'
        if group == 'total':
            s = 'Life'
            
        plt.title('Times Through ' + s + ' Penalty' + adj)
    
        right = sum(weights > 1000) + 1.5
        y = y[:int(right-.5)]
        
        if right < 6:
            left = 0.5
        else:
            left = 0
        
        plt.xlim((left, right))
        plt.ylim((0.27, 0.38))
        
        if trendline:
            p = (' points ' if round(1000*slope) > 1 else ' point ')
            ax.text(left + (right-left)*0.1,  0.358, '+' + str(round(1000*slope)) + p + 'of wOBA per matchup')
        
        plt.show()

    return slope, intercept



#
# Find and adjust for TTO penalty
#

m, b = calculate_penalty(matchup_df, 'game', trendline=True)

matchup_df['woba_adj_tto'] = matchup_df['woba_value'] - matchup_df['matchup_game']*m



#
# TTS penalty
#

m, b = calculate_penalty(matchup_df, 'season', trendline=True, adjusted=True)



#
# TTL penalty
#

m, b = calculate_penalty(matchup_df, 'total', trendline=True, adjusted=True)



#
# First PA penalty by long-term familiarity
#

x = np.arange(1,16)
y = np.empty(15)
weights = np.empty(15)
for n in x:
    fam_df = matchup_df[matchup_df['matchup_game_number'] == n]
    first_df = fam_df[fam_df['matchup_game'] == 1]
    weights[n-1] = len(first_df)
    y[n-1]  = first_df['woba_value'].mean()

# regression
slope, intercept, r_squared = weighted_linear_regression(x, y, weights)

# plot
fig, ax = plt.subplots(figsize=(8, 5))    
x_line = np.array([min(x), max(x)])
y_line = slope * x_line + intercept

plt.plot(x_line, y_line, 'k--')

# points with weight
max_size = 1500
scale = weights[0]/max_size
scaled_sizes = weights/scale
plt.scatter(x, y, s=scaled_sizes, alpha=1)

# formatting
plt.xlabel('Number of Games Facing this Batter')
plt.ylabel('wOBA')
    
plt.title('Long-Term Familiarity - First Plate Appearance of Nth Game')

right = sum(weights > 1000) + 1.5
y = y[:int(right-.5)]

if right < 6:
    left = 0.5
else:
    left = 0

plt.xlim((left, right))
plt.ylim((0.27, 0.4))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
p = (' points ' if round(1000*slope) > 1 else ' point ')
ax.text(left + (right-left)*0.1,  0.358, '+' + str(round(1000*slope)) + p + 'of wOBA per game')

plt.show()



#
# TTO by long-term familiarity
#

x = np.arange(1,16)
y = np.empty(15)
weights = np.empty(15)
for n in x:
    fam_df = matchup_df[matchup_df['matchup_game_number'] == n]
    weights[n-1] = len(fam_df)
    y[n-1], b  = calculate_penalty(fam_df, 'game', plot=False)

# regression
slope, intercept, r_squared = weighted_linear_regression(x, y, weights)

# plot
fig, ax = plt.subplots(figsize=(8, 5))    
x_line = np.array([min(x), max(x)])
y_line = slope * x_line + intercept

plt.plot(x_line, y_line, 'k--')

# points with weight
max_size = 1500
scale = weights[0]/max_size
scaled_sizes = weights/scale
plt.scatter(x, y, s=scaled_sizes, alpha=1)

# formatting
plt.xlabel('Number of Games Facing this Batter')
plt.ylabel('TTO Effect (wOBA change per TTO)')
    
plt.title('TTO Penalty by Long-Term Familiarity')

right = sum(weights > 1000) + 1.5
y = y[:int(right-.5)]

if right < 6:
    left = 0.5
else:
    left = 0

plt.xlim((left, right))
plt.ylim((0.005, 0.025))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
p = (' points ' if round(1000*slope) > 1 else ' point ')
ax.text(left + (right-left)*0.1,  0.358, '+' + str(round(1000*slope)) + p + 'of wOBA per matchup')

plt.show()



#
# Long and short term together
#

x = np.arange(0.667,7.333, 0.333)
y = np.empty(21)
weights = np.empty(21)
for n in tqdm(np.arange(1,8)):
    fam_df = matchup_df[matchup_df['matchup_game_number'] == n]
    for t in range(1,4):
        if t < 3:
            tto_df = fam_df[fam_df['matchup_game'] == t]
        else:
            tto_df = fam_df[fam_df['matchup_game'] >= t]
            
        weights[(n-1)*3 + t - 1] = len(tto_df)
        y[(n-1)*3 + t - 1] = tto_df['woba_value'].mean()


def format_decimal(num):
    
    # Get the original string format to preserve trailing zeros
    # For negative numbers
    if num < 0:
        
        num = round(num, 3)
        str_num = f"{num:f}"
        # Remove the '0' after the minus sign but keep everything else
        return '-' + str_num[2:6]

    num = round(num, 3)
    str_num = f"{num:f}"
    # For positive numbers, just remove the leading '0'
    return '+' + str_num[1:5]


# plot
fig, ax = plt.subplots(figsize=(8, 5))    

# points with weight
max_size = 1500
scale = weights[0]/max_size
scaled_sizes = weights/scale
plt.scatter(x, y, s=scaled_sizes, alpha=1)

# formatting
plt.xlabel('Number of Games Facing this Pitcher', fontsize=14)
plt.ylabel('wOBA', fontsize=14)
plt.title('Short-Term and Long-Term Familiarity Penalties', fontsize=16)

right = sum(scaled_sizes > 1)/3 + 0.5
left = 0.2

plt.xlim((left, right))
plt.ylim((0.27, 0.4))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

# trendlines
n_groups = int(np.ceil(max(x) - min(x)))


# For each group
for i in range(n_groups):
    # Calculate group boundaries
    start = min(x) + i - 0.01
    end = start + 0.677
    
    # Get points in this group
    mask = (x >= start) & (x < end)
    group_x = x[mask]
    group_y = y[mask]
    
    if len(group_x) > 1:  # Need at least 2 points for a trendline
        # Calculate trendline
        slope, intercept, r_value, p_value, std_err = stats.linregress(group_x, group_y)
        
        # Create trendline points
        line_x = np.array([min(group_x), max(group_x)])
        line_y = slope * line_x + intercept
        
        # Plot trendline
        plt.plot(line_x, line_y, color='red', linestyle='dashed')
        
        s = format_decimal(slope/3)
        plt.text(i+1, group_y.max() + 0.005, s, ha='center', color='red')


# For each first PA
for i in range(n_groups - 1):
    
    start = i*3
    end = (i+1)*3
        
    group_x = np.array([x[start], x[end]])
    group_y = np.array([y[start], y[end]])
    
    if len(group_x) > 1:  # Need at least 2 points for a trendline
        # Calculate trendline
        slope, intercept, r_value, p_value, std_err = stats.linregress(group_x, group_y)
        
        # Create trendline points
        line_x = np.array([min(group_x) + 0.1, max(group_x) - 0.15])
        line_y = slope * line_x + intercept - 0.01
        
        # Plot trendline
        plt.plot(line_x, line_y, color='orange', linestyle='dashed')
        
        s = format_decimal(slope)
        plt.text(i+1, line_y.min() - 0.01, s, color='orange')
            

# add legend
from matplotlib.lines import Line2D

legend_elements = [Line2D([0], [0], lw = 4, color='red', label='Times Through the Order'),
                   Line2D([0], [0], lw = 4, color='orange', label='First PA After N Games')]
                          
# Create the figure
ax.legend(handles=legend_elements, loc=(0.05,0.75), title = 'wOBA penalty for...', title_fontsize=12)

# Add secondary x-axis for PA labels
ax2 = ax.secondary_xaxis("bottom")  # Create secondary axis at bottom

ax2.xaxis.set_ticks_position("bottom") # Added this line
ax2.xaxis.set_label_position("bottom") # Added this line
ax2.tick_params(direction='in', pad = -15)

# Calculate tick positions for repeated 1,2,3+ labels
game_positions = np.arange(1, 8)
pa_positions = []
pa_labels = []

# Create PA ticks centered under each game number
for game in game_positions:
    # Create three evenly spaced ticks under each game number
    pa_positions.extend([game - 0.2, game, game + 0.2])
    pa_labels.extend(['1', '2', '3+'])

# Set the tick positions and labels for both axes
ax2.set_xticks(pa_positions)  # PA labels on bottom axis
ax2.set_xticklabels(pa_labels, fontsize=8)  # Make PA labels slightly smaller to avoid crowding

# Set axis labels
ax.set_xlabel('Number of Games Facing this Batter')
ax.tick_params(axis='x', labelsize=14, width=0, length=5)
ax2.set_xlabel('Times Through the Order', labelpad = -30)

plt.tight_layout()
plt.savefig('/Users/johnnynienstedt/Library/Mobile Documents/com~apple~CloudDocs/Baseball Analysis/Pitching Analysis/TTO Analysis/TTO_by_fam_game_level.png', dpi=600)

plt.show()

















# first PA penalty for same week
first_df = matchup_df[matchup_df['matchup_game'] == 1]


def analyze_weekly_matchups(data):
    """
    Analyze batter-pitcher matchups that occur within the same week.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns 'game_date', 'batter', 'pitcher'
    
    Returns:
    pandas.DataFrame: DataFrame containing matchup information
    """
    # Convert game_date from string to datetime
    df = data.copy()
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Add week end date 
    df['week_end'] = df['game_date'] + 7
    
    # mark all p-b matchups within this week
    
    # Group by week, batter, and pitcher
    weekly_matchups = df.groupby(['week_start', 'batter', 'pitcher']).agg({
        'game_date': ['count', 'min', 'max']  # Count of matchups and date range
    }).reset_index().copy()
    
    # Flatten column names
    weekly_matchups.columns = ['week_start', 'batter', 'pitcher', 
                             'matchup_count', 'first_matchup', 'last_matchup']
    
    # Filter for multiple matchups in the same week
    multiple_matchups = weekly_matchups[weekly_matchups['matchup_count'] > 1].copy()
    
    # Calculate days between first and last matchup
    multiple_matchups['days_between'] = (multiple_matchups['last_matchup'] - 
                                       multiple_matchups['first_matchup']).dt.days
    
    # Sort by week and matchup count
    multiple_matchups = multiple_matchups.sort_values(['week_start', 'matchup_count'], 
                                                    ascending=[False, False])
    
    return multiple_matchups

mm = analyze_weekly_matchups(first_df)





















import pandas as pd

def analyze_rolling_matchups(df):
    """
    Analyze batter-pitcher matchups using a 7-day rolling window from each game date.
    Creates a unique identifier for each batter-pitcher-window combination.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns 'game_date', 'batter', 'pitcher'
    
    Returns:
    pandas.DataFrame: Original DataFrame with additional columns for analysis
    """
    # Convert game_date from string to datetime (if not already)
    if df['game_date'].dtype == 'object':
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'].str.replace('_', '-'))
    
    df[['batter', 'pitcher']] = df[['batter', 'pitcher']].astype(str)
    
    # Add week_end column (7 days after each game)
    df['week_end'] = df['game_date'] + pd.Timedelta(days=7)
    
    # Sort by date for efficient processing
    df = df.sort_values('game_date')
    
    # Create a unique matchup string
    df['matchup'] = df['pitcher'] + '_' + df['batter']
    
    # Initialize the window identifier column with NaN
    df['window_id'] = pd.NA
    
    def assign_window_ids(group):
        """Helper function to process each batter-pitcher matchup group"""
        current_window_start = None
        current_window_id = None
        
        # Convert to numpy for faster operations
        dates = group['game_date'].values
        
        # Initialize window_ids array
        window_ids = []
        
        for date in dates:
            if current_window_start is None or date > current_window_start + pd.Timedelta(days=7):
                # Start new window
                current_window_start = date
                current_window_id = f"{group['matchup'].iloc[0]}_{date.strftime('%Y%m%d')}"
            window_ids.append(current_window_id)
        
        return pd.Series(window_ids, index=group.index)
    
    # Apply the window assignment function to each matchup group
    df['window_id'] = df.groupby('matchup', group_keys=False).apply(assign_window_ids)
    
    # Add count of games in each window
    df['games_in_window'] = df.groupby('window_id')['game_date'].transform('count')
    
    # Sort by date and matchup for cleaner output
    df = df.sort_values(['game_date', 'matchup'])
    
    return df

def get_window_summary(df):
    """
    Create a summary of all windows that contain multiple games.
    
    Parameters:
    df (pandas.DataFrame): DataFrame output from analyze_rolling_matchups
    
    Returns:
    pandas.DataFrame: Summary of windows with multiple games
    """
    summary = df[df['games_in_window'] > 1].groupby('window_id').agg({
        'matchup': 'first',
        'game_date': ['min', 'max', 'count'],
        'pitcher': 'first',
        'batter': 'first'
    }).reset_index()
    
    summary.columns = ['window_id', 'matchup', 'first_game', 'last_game', 'games_in_window', 
                      'pitcher', 'batter']
    
    summary['days_between'] = (summary['last_game'] - summary['first_game']).dt.days
    
    return summary.sort_values(['first_game', 'matchup'])


# Get detailed data with window assignments
detailed_df = analyze_rolling_matchups(first_df)

# Get summary of windows with multiple games
summary = get_window_summary(detailed_df)