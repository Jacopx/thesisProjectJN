Hadoop Data Analysis
------------------

The data structure is composed of:
  # hadoop_issue.csv:
    * issue_id: Reference to sourcing platform ID
    * type: Type of the issue [Bug, Improvement, SubTask, New F, Test, Task, Wish]
    * Priority: The importance of the issue opened [Blocker, Critical, Major, Minor, Trivial]
    * create_date: datetime of issue open
    * resolved_date: datetime of issue closing

  # hadoop_issue-modification.csv:
    * issue_id: Reference to sourcing platform ID
    * added_lines: Aggregation, respect multiple commit, of the number of line added
    * removed_lines: Aggregation, respect multiple commit, of the number of line removed


To forecast
------------
Every analysis performed over this dataset has not yet been published. Here some ideas:
  - Issue over time
  - Issue prediction after commit


Analysis
---------
The case seems to not be study yet, in any case, it can be categorized as a generical forecasting problem. After some research, the LSTM, Long Short-term Memory Neural Network, seems to be a good model to be implemented to study this case.


External Research
------------------
* [General TS forecasting] https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/
* [General TS forecasting] https://www.kdnuggets.com/2019/05/machine-learning-time-series-forecasting.html
* [Long Short Term Memory] https://www.kdnuggets.com/2017/10/guide-time-series-prediction-recurrent-neural-networks-lstms.html
* [General TS forecasting] https://www.hindawi.com/journals/complexity/2019/9067367/
