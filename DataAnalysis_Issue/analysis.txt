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
Issue over time, there is of course a correlation respect the day of the week, saturday and sunday report really few issue. Is also interesting to notice that, after a commit for a solved issue, other problem arise. Many issue means many code lines to solve them.


Important Feature
-----------------
  * Number of issue
  * Create date


Analysis
---------
The case seems to not be study yet, in any case, it can be categorized as a generical forecasting problem. After some research, the LSTM, Long Short-term Memory Neural Network, seems to be a good model to be implemented to study this case.


External Research
------------------
* [General TS forecasting] https://machinelearningmastery.com/how-to-develop-deep-learning-models-for-univariate-time-series-forecasting/
* [General TS forecasting] https://www.kdnuggets.com/2019/05/machine-learning-time-series-forecasting.html
* [Long Short Term Memory] https://www.kdnuggets.com/2017/10/guide-time-series-prediction-recurrent-neural-networks-lstms.html
* [General TS forecasting] https://www.hindawi.com/journals/complexity/2019/9067367/
