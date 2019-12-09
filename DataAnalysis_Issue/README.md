SEOSS33 Data Analysis
------------------

The whole data structure and other infos can be found in the reference paper [https://www.researchgate.net/publication/333347361_The_SEOSS_33_Dataset_-_Requirements_Bug_Reports_Code_History_and_Trace_Links_for_Entire_Projects]


To forecast
------------
Every analysis performed over this dataset has not yet been published. Here some ideas:
  - Issue over time
  - Issue prediction after commit

After a lot of research, the new target, is to predict the severity of the issue that will be opened in the future,
this task require extraction of a lot of different features. More information will be available in the thesis document. 


Analysis
---------
The case seems to not be study yet, in any case, it can be categorized as a generical forecasting problem. After some research, the LSTM, Long Short-term Memory Neural Network, seems to be a good model to be implemented to study this case.


External Research
------------------
* [General TS forecasting] https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/
* [General TS forecasting] https://www.kdnuggets.com/2019/05/machine-learning-time-series-forecasting.html
* [Long Short Term Memory] https://www.kdnuggets.com/2017/10/guide-time-series-prediction-recurrent-neural-networks-lstms.html
* [General TS forecasting] https://www.hindawi.com/journals/complexity/2019/9067367/
