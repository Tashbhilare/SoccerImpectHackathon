# Bundesliga Player Valuation System
## Soccer Data Analytics Hackathon 2026

### Team Members
- Tanish Bhilare
- Mohit Sharma

### Project Overview
Building a transparent, interpretable player valuation metric using IMPECT Open Data from the 2023/24 Bundesliga season.

**Prompt:** (B) Build a Transparent Player Valuation Metric

### Approach
Developing a multi-dimensional rating system with 8 granular metrics:
- **Attacking:** Finishing, Chance Creation, Dribbling
- **Defensive:** Ball Winning, Defensive Actions
- **Passing:** Passing Accuracy, Ball Progression, Long Passing

### Progress (Checkpoint - Feb 9)
- **Data Pipeline:** Successfully loaded 962,990 events from 306 matches
- **Event Weighting System:** Defined position-aware weights for attack/defense/passing
- **Player Ratings:** Generated ratings for 402 players across 3 positions
- **Role-Specific Metrics:** Implemented granular metrics tailored to positions

### Next Steps
- [ ] Refine zone classification (team-aware implementation)
- [ ] Create visualizations (player leaderboards, radar charts)
- [ ] Select case study player
- [ ] Validate against ground truth (goals, assists)
- [ ] Build final presentation

### Tech Stack
- Python 3.11
- kloppy (soccer data processing)
- polars (high-performance dataframes)
- numpy, scikit-learn

### Repository Structure
- `notebooks/1_data_loading.ipynb` - Loads and processes IMPECT Open Data
- `notebooks/2_event_scoring.ipynb` - Calculates event-based player ratings
- `notebooks/3_role_specific_ratings.ipynb` - Computes granular role-specific metrics

### Data Source
IMPECT Open Data - German Bundesliga 2023/24 season (306 matches)
```

3. **Create requirements.txt:**
```
kloppy>=3.18.0
polars>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
requests>=2.31.0
tqdm>=4.66.0
