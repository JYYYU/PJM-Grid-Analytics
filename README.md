# Grid Analytics

**PJM Energy Market Analysis and Forecasting**

This project analyzes historical grid performance data in the PJM energy market to understand price variability, outage impacts, and grid stress. By leveraging advanced data science techniques and machine learning models, this project aims to uncover key insights and predict critical grid conditions.


**Project Highlights:**
1. Historical Analysis:
- Understand LMP (Locational Marginal Price) trends and variability across nodes and regions.
- Examine planned vs. forced outages and their influence on grid performance.

2. Risk Identification:
- Identify regions prone to congestion using metrics like LMP volatility, outage intensity, and capacity margin.
- Highlight stress indicators to detect early signs of emergency conditions.

3. Regional Comparisons:
- Compare grid performance across zones, pinpointing disparities and stability.

4. Machine Learning Predictions:
- Develop predictive models for emergency-triggered conditions, congestion risks, and LMP price variability.
Use state-of-the-art algorithms like Decision Trees, Random Forests, and Gradient Boosting for enhanced accuracy.

5. Actionable Dashboard:
- Deliver an interactive visualization platform to support maintenance planning, infrastructure upgrades, and grid reliability improvements.

**Limitations:** Weather Data Constraints
One key limitation in this analysis is the lack of granular weather data, which could enhance the accuracy of congestion risk and price forecasting. 

Factors such as:

- Wind speeds and direction (critical for renewable generation forecasts)
- Temperature and humidity (major drivers of demand fluctuations)
- Rainfall and storms (which can impact outages and transmission performance)
- Snow and ice accumulation (which can lead to winter-related reliability challenges)
  
were not included due to data sourcing challenges. Given the strong correlation between extreme weather events and grid stress, incorporating high-resolution weather data in future iterations could significantly improve forecasting accuracy.


**Why This Matters:**
As the PJM grid faces rising demand and increasing complexity, this project equips energy stakeholders with the tools to predict, analyze, and mitigate risks. From identifying high-risk nodes to optimizing grid reliability, the insights from this analysis can drive better decision-making and improve market outcomes.

# Acknowledgments:
This project utilized PJM Data Miner 2 for accessing historical grid data. The following resources were instrumental in helping understand and use the API:

- Robert Zwink DataMiner GitHub Repository (https://github.com/rzwink/pjm_dataminer)
- PJM Data Miner 2 API Guide (https://www.pjm.com/-/media/etools/data-miner-2/data-miner-2-api-guide.ashx)

A huge thanks to the contributors of these resources for their work in simplifying the data access process.



