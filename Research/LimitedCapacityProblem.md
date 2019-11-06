## Limited Capacity Problem

### Paper
Each paper added to the repository will be analysed and summarised to make the search process easier:
  * **Bike Sharing**:
    * **BikeAvail_ML**: Using RF, LSBoost and PLSR to model bike available at each station. Use also near stations to make prediction.
    * **DemandPrediction_BS**: Hierarchical traffic prediction model to predict check-out/in number of each station cluster. Add station clustering with iterative spectral clustering algorithms. Use GBRT to predict total check-out/in.
    * **BikeNumber**: Mathematical analysis (using BGIP) to predict number of bikes at each station.
    * **BikeForecasting_for_Rebalancing**: Full-stack system to optimise daily rebalancing operations, of BS systems, using MLP and RF algorithms.
    * **PredictingOccupancyTrends**: Saturation prediction of Barcelona, prediction, up to 2 days, of full or empty station.
    * **BalanceBS_System**: Analyse user behaviour in BSS to improve the quality of it, both in case of incentive or not. Use clustering on bike rides.
    * **OptimizationFieldOp_BS**: Full-stack system to optimised BSS management. Prediction studied with MLP algorithm.
    * **PredictOneDayAhead**: Analysis at City level of granularity, different algorithm and considerations about over-fitting problems
    * **CloseLoop_PredictionBS**: Use probabilistic station-station approach. Based on large Hangzhou dataset, manage also unpredicted events. Evaluate a lot of prediction algorithm like baseline: HM, ARMA, RF, PFM, PD.

  
  * **Analog Problem**:
    * **OilSaturation**: Paper over oil saturation prediction, based on LM Neural Network, to reduce cost of experimental analysis.
    * **UrbanPassenger_SVN**: Forecasting of passenger demand over public urban transport based on SVM.
    * **ArrivalTimePrediction**: Live forecasting of arrival time of urban transportation systems with different techniques, historical, statistical, ML and hybrid models.
    * **CityTrafficPrediction**: Two models, one based on traffic flow propagation and the other one on spare road capacity.
    * **UrbanSaturationForecast**: Forecasting urban power load based on saturated load density curve. Not ML algorithms.
    * **HeatLoad_ML**: Forecasting power demand using for 4 algorithms: SVR, RT, FFNN and MLR.
    * **Railway_RL**: Timetable scheduling using RL.
    * **HosDischarge**: Predict impatient discharge to improve management and allocation of hospital bed resources. Prediction performed using ARIMA, LSTM and RF.
    * **HosPredictionQualityIndicator**: No AI, only optimisation methods.


### Analysis
A lot of paper are found about the BSS problems, the great majority are related to the prediction of saturation situations in order to improve the quality of bike redistribution. Several different approaches are used, from the clustering of the bike stations used to take in account neighbour stations during prediction, to the clustering of bike rider. The forecasting horizon is discussed, most of them varying from 15 minutes to some hours, some of them also to more days. In all system the weather as been taken into account, due to its nature the bike rider are really dependent on this feature. Results are similar and also the algorithms used are the same, RF seems to be the top scorer. The difference of feature extracted do not require modifications over the algorithms.
Other realities, with similar saturation problems, use BP algorithm, lot of studies show that also other strategies can be used to improve the quality of prediction, like SVM.
Take into account the near past seems to be a good strategy for all kind of different topic.

The common path of all the analysis is to evaluate relation between events and first, its geospatial position and second, the time position; for example, stations clustering for bikes, time of the day for rides, etc...

The most used strategies is the prediction using Random Forest, it seems to be the one with the lower mean error.


### Acronymous
  * **BS**: Bike Sharing
  * **BSS**: Bike Sharing Systems
  * **RF**: Random Forest
  * **LSBoost**: Least-Squares Boosting
  * **PLSR**: Partial Least-Squares Regression
  * **MLP**: Multi Layer Percepton
  * **GBRT**: Gradient Boosting Regression Tree
  * **BGIP**: Bimodal Gaussian Inhomogeneous Poisson
  * **BRP**: Bike Repositioning Problem
  * **HM**: Historical Mean
  * **ARIMA**: AutoRegressive Integrated Moving Average
  * **PFM**: Probabilistic Flow Model
  * **SVM**: Support Vector Machine
  * **PD**: Potential Demand
  * **BP**: Back Propagation
  * **NN**: Neural Network
  * **LM**: Levenberg-Marquardt
  * **ML**: Machine Learning
  * **ITS**: Intelligent Transportation System
  * **SVR**: Support Vector Regression
  * **RT**: Regression Tree
  * **FFNN**: Feed Fprward NN
  * **MLR**: Multiple Linear Regression
  * **RL**: Reinforcement Learning
  * **LSTM**: Long Short-Term Memory
