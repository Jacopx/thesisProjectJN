VVF Data Analysis
------------------

The data structure is composed of:
  # year.csv:
    * n: number of operation respect year
    * date: YYYY-MM-DD of the operations
    * start: start time of the operation
    * finish: end time fo the operation
    * duration: [s] of duration of the operation
    * [x,y]: GPS localization
    * loc: Village
    * typo: Typology of the operation


To forecast
------------
It could be interesting search the period with low or high number of operations, some example are:
  - Bonifica insetti = during summer
  - Bonifica = n
  - Mesi = n
  - Alberi Pericolanti = summer
  - Dissesto Statico = November
  - Incidente Stradale = Saturday
  - Incidente Stradale = 8/17

Forecasting also the number of generic operations is important, accordingly to the day hour.


Important Feature
-----------------
  * Typology of operation
  * Date
  * Start time
