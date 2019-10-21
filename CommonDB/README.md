CommonDB
-----------
The goal of this structure is to create a common place to store the data independently of the kind/source of it.
Having a common structure is useful to make the procedure of forecasting standard and get it always in the same way.
The previous version v1 had the problem to be too much generic, the 'Capacity Management Problem' (CMP) require a
more precise and fitted data model in order to correctly achieve the goal.


Structure
----------
Is composed of five tables:
* **event**: ***eid, dataset***, etype, start_dt, end_dt
* **involved**: ***eid, dataset, type, id***
* **object**: ***id, dataset***, type
* **info**: ***id, dataset***, type, description
* **location**: ***id, dataset***, latitude, longitude

The main table is EVENT, for each event there are one or more object involved and the table involved is used to link this relationship.
Info and Location are used to store more information about each object. Everything is enforced by some key=foreign_key pair.


Utilization
------------
The process is based on the files contained in the folder 'data_to_load':
* ***data.csv***: Is the file with the data to be loaded in the system
* ***dict.txt***: [system_db:csv_field] Is a file of used to match the field of the DB with the field of the CSV

The usage of dict is upgraded respect v1.0, now also conditional mapping is available.
For example: we need to map 3 different object, the start station, the ending station and the bike used.
Both stations require INFO and LOCATIONS entries, the bike is only an object. Mapping each feature allow the
creation of multiple tables for each object, adding only 2 of them, will prevent the creation of one table; set
all the values to NONE will make an object without tables linked.

The system can also prevent to adding duplicate. For example: the destination of an operations is repeater for a
number of times equals to the number of unit involved, setting the value to **ONE** force the system to add this entry
only one time, the unique name will be composed by a prefix, a dash and the eid string. In case of **UNIQUE**
the system will add more time this feature and make an unique key for them, this function is useful when is not possibile
to defined automatically an unique name.
In case, for the same dataset, some collision can occur, like bike_id and station_id, a specific suffix adder must be created inside the `main.py`


Logical constraints
--------------------
Not all kind of datasets can be uploaded into the system.
The system is developed to store dataset based on events, an event is a portion of time that can be categorized in someway,
the event can be also instantaneous. If a dataset is event based it can be loaded in the system, otherwise is impossible.
Event like log report can be also loaded, the duration can be computed and the 'start_datetime' become the time of the event execution.
Each event can involve multiple object, of different type, the must be linked to the event with a relationship,
save in the 'type' field of the involved table. An event could not have object involved, this could drive to loose the focus of the
structure, but is feasible. Also loading a NON TIME BASED events dataset is possible, but it loose the goal of the structure.
In general every dataset can be fit to this structure but not all of them could gain advantages by this solution, in some case
could become only an interactive big CSV.


Changelog
----------
Only major release, minor are part of the same major.
- v2.4: **[LATEST]** Allow upload of saturation data
- v2.3: Suffix implementation
- v2.1: Working solution with some collision due to duplicated object_id
- v2.0: First complex implementation, to simple, missing dataset reference and some type
- v1.0: Super simple match solution, similar to CSV on web
