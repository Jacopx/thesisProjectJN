Forecast Analysis
------------------
The purpose of this script is to perform analysis over the datasets loaded in CommonDB of via CSV files.


Functions
----------
* **Distance**: Calculate geodesic distance from source and destination point of events
* **Duration**: Calculate duration of events
* **Saturation**: Calculate amount of minute in saturation situations
* **Forecast**: Make forecast prediction in three different ways:
  * **RandomForest**: Classic prediction algorithm of SkLearn
  * **NeuralNetwork**: Personal NN developed with Keras
  * **LongShortTermMemory**: NN with LSTM layers
