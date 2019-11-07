## Fit problem to DEV scenario
The goal of this research is to get similarities between different dataset and the Development one. The focus is to conduct similar analysis to evaluate if the generalisation of the data still allow to get sensitive information despite differences.

### DEV Scenario
The DEVELOPMENT dataset is based on issue, each issue can be closed or still open, each issue can have a commit related, each commit is done by a developer. The main features are:
  * Limited number of resources: Developers
  * Event: issue, commit
  * Event correlation: Commit --> Issue

### Paper
Analysis pro and cons of each paper:
  * **BIKE SHARING**:
    * **BikeAvail_ML**: [DATA] Add the spatial correlation of stations and weather evaluation, is not possible to spatial correlating developers, neither using weather.
    * **DemandPrediction_BS**: [DATA*] Add stations clustering and weather evaluation, same problem reported above.
    * **BikeNumber**: [DATA] Associate the C_IN/C_OUT with the Poisson distribution.
    * **BikeForecasting_for_Rebalancing**: [DATA*] Add stations clustering, holidays and weather evaluation, same problem reported above.
    * **PredictingOccupancyTrends**: [DATA*] Make use of weather evaluation, holidays and select station, to reduce dataset, based on special Barcelona features.
    * **BalanceBS_System**: [DATA*] Take into account reward system to mitigate saturation situations. Clustering of rides by customer class.
    * **OptimizationFieldOp_BS**: [DATA*] Neighbour station, weather and holidays.
    * **PredictOneDayAhead**: [DATA*] City granularity level, use also weather.
    * **CloseLoop_PredictionBS**: [DATA*] Consider weather, online usage counter and takes into account unpredicted event (concert, etc...).


  * **OTHER**:
    * **OilSaturation**: [NO DATA] Physical features, no units or discrete resource.
    * **UrbanPassenger_SVM**: [DATA] Yearly aggregated data, passenger count, no capacity planning and no resource to manage.
    * **ArrivalTimePrediction**: [NO DATA] Arrival time prediction based on cities specific features, like weather, traffic, driving dynamics, etc... Resources not taken into account.
    * **CityTrafficPrediction**: [NO DATA] Research paper about traffic study, no technical implementations.
    * **UrbanSaturationForecast**: [NO DATA] Based on population growth and architectural planning.


The * on the data means that is similar data but from a different locations, project or structure.
