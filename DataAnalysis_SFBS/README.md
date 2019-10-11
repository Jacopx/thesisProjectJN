SF BikeSharing Data Analysis
------------------
The data getted from: https://www.kaggle.com/benhamner/sf-bay-area-bike-share
The data structure is composed of:
  * station.csv - Contains data that represents a station where users can pickup or return bikes.
  * status.csv - data about the number of bikes and docks available for given station and minute.
  * trips.csv - Data about individual bike trips
  * weather.csv - Data about the weather on a specific day for certain zip codes


To forecast
------------
* Already studied:
  - Bike availability respect actual time and specific stations

* Not yet studied:
  - Free dock slots availability respect actual time
  - Forecast number of active rides
  - Duration of a ride
  - Destination prediction
  - Resizing of docks
  - Bike necessary to accomodate ride request


Analysis
---------
All the previuos research about this, or similar, dataset show a lot of different correlation. One of the most important is the type of client (customer or subscriber) and the time, or the day of the week, of the utilization. There are also, in case of SF, lot of correlation with the weather.
Searching through previous experience seems that RNN, like LSTM, could achieve the goal of this research.


External Research
------------------
* [SimpleDeterministic, Random Forest, RF+Weather] https://towardsdatascience.com/sf-bike-share-predictions-e15316ce300f
    ** [Code] https://github.com/gussie/Springboard_data_science
    ** [Data] https://www.kaggle.com/benhamner/sf-bay-area-bike-share

* https://www.simplydatascience.it/bike-sharing-analysis/

* [RandomForest] https://www.manishkurse.com/PythonProjects/SFO_BikeShare.html
* [RandomForest] https://www.manishkurse.com/PythonProjects/BikeSharingDemand.html

* [Max Likelihood Estimate (MLE), Max A Posteriori (MAP)] https://towardsdatascience.com/predicting-no-of-bike-share-users-machine-learning-data-visualization-project-using-r-71bc1b9a7495

* [Public Tableau] https://public.tableau.com/views/SFBikeShareAnalysis/FinalPresentation?:embed=y&:showVizHome=no&:display_count=y&:display_static_image=y&:bootstrapWhenNotified=true

* [Only Analysis] https://datascienceplus.com/exploring-san-francisco-bay-areas-bike-share-system/

* https://towardsdatascience.com/traditional-vs-deep-learning-algorithms-in-retail-industry-i-b7b7f86793d4

* [Forecasting with LSTM] https://towardsdatascience.com/time-series-analysis-visualization-forecasting-with-lstm-77a905180eba

* [LSTM] https://medium.com/google-cloud/predicting-san-francisco-bikeshare-availability-with-tensorflow-and-lstms-a3ced14d13dc


Paper
--------
* ReplenInventory.pdf
* InvMan_BigData.pdf
* AI_StockDecision.pdf
* StatAndML_MethodsForecasting.pdf
* BikesAvailabilityinaBike-SharingML.pdf
